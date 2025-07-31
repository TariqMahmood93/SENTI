
import os
import glob
import re
import random
import logging
from copy import deepcopy
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformers import (RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification, RobertaTokenizer, AutoTokenizer, AutoModel)
from logging_customized import setup_logging
from config import read_arguments, Config
from torch_initializer import initialize_gpu_seed
from data_loader import DataType, load_data
from data_representation import BERTProcessor
from optimizer import build_optimizer
from evaluation import Evaluation


#--------------------------------Dynamic imputation for multiple percentages of nulls---------------------------------------------

PCT_NULLS = [5,10,20,40]
SEED      = 1234


dataset = "adultsample"
training = "Retraining"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("CUDA available:", torch.cuda.is_available())
setup_logging()
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ---- MEAN‐POOLING for LaBSE ----
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# ---- ATTRIBUTE DICTIONARY ----
ATTR_DICT = {
    'adultsample': ['age','workclass','education','education-num','marital-status', 'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income'],
    # 'australian':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15'],
    # 'contraceptive':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'],
    # 'credit':['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16'],
    # 'imdb':['color', 'director', 'actor_2', 'actor_1', 'title', 'actor_3', 'original_language', 'production_countries', 'content_rating', 'year', 'vote_average'],
    # 'mammogram':['A1', 'A2', 'A3', 'A4', 'A5'],
    }

# ---- DATA GENERATOR ----
class CategoryDataGenerator:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path, dtype='str')

    def get_truth_data(self, cat_attr: str) -> pd.DataFrame:
        truth = self.data[self.data[cat_attr].notna()]
        cols = [c for c in self.data.columns if c not in ['id', cat_attr]]
        truth['text_a'] = truth[cols].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        truth = truth[['id','text_a',cat_attr]].rename(columns={cat_attr:'label'})
        truth['label'] = truth['label'].apply(lambda x: self.value_to_ix[cat_attr][x])
        return truth

    def get_mis_data(self, cat_attr: str) -> pd.DataFrame:
        mis = self.data[self.data[cat_attr].isna()]
        cols = [c for c in self.data.columns if c not in ['id', cat_attr]]
        mis['text_a'] = mis[cols].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        return mis[['id','text_a']]

    def get_train_data(self, dataset_name: str, ignore_attr=['id'], training_ratio=1.0):
        attrs = ATTR_DICT[dataset_name]
        self.value_to_ix, self.ix_to_value = {}, {}
        out = {}
        for a in attrs:
            domain = set(self.data[a].dropna())
            v2i = {v:str(i) for i,v in enumerate(domain)}
            i2v = {str(i):v for i,v in enumerate(domain)}
            self.value_to_ix[a], self.ix_to_value[a] = v2i, i2v
            td = self.get_truth_data(a).sample(frac=training_ratio, random_state=SEED)
            split = int(len(td)*0.7)
            out[a] = (
                td.iloc[:split].reset_index(drop=True),
                td.iloc[split:].reset_index(drop=True)
            )
        return out

    def get_impute_data(self, dataset_name: str, ignore_attr=['id']):
        return {a:self.get_mis_data(a) for a in ATTR_DICT[dataset_name]}

# ---- TRAIN & IMPUTE FUNCTIONS (unchanged) ----
def train_bert(device, train_loader, model, tokenizer, optimizer, scheduler,
               evaluation, num_epochs, max_grad_norm, model_type):
    model.zero_grad()
    best_f1, best_model = -1, model
    for _ in trange(int(num_epochs), desc="Epoch"):
        for batch in tqdm(train_loader, desc="Iteration"):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':batch[0],'attention_mask':batch[1],'labels':batch[3]}
            if model_type!='distilbert':
                inputs['token_type_ids'] = batch[2] if model_type in ['bert','xlnet'] else None
            loss = model(**inputs)[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step(); scheduler.step(); model.zero_grad()
        res = evaluation.evaluate(model, device, None)
        if res['f1_score'] > best_f1:
            best_f1, best_model = res['f1_score'], model
    return best_model

def train(model, tr_exs, vl_exs, tokenizer, device, label_list, args):
    tr_loader = load_data(tr_exs, label_list, tokenizer,
                          args.max_seq_length, args.train_batch_size,
                          DataType.TRAINING, args.model_type)
    num_steps = len(tr_loader) * args.num_epochs
    optimizer, scheduler = build_optimizer(
        model, num_steps,
        args.learning_rate, args.adam_eps,
        args.warmup_steps, args.weight_decay
    )
    ev_loader = load_data(vl_exs, label_list, tokenizer,
                          args.max_seq_length, args.eval_batch_size,
                          DataType.EVALUATION, args.model_type)
    evaluation = Evaluation(ev_loader, None, None, len(label_list), args.model_type)
    return train_bert(device, tr_loader, model, tokenizer,
                      optimizer, scheduler, evaluation,
                      args.num_epochs, args.max_grad_norm, args.model_type)

def impute_category(data, model, tokenizer, device, args):
    if data.empty:
        return np.array([], dtype=int)
    model.eval()
    vocab = tokenizer.get_vocab()
    proc  = BERTProcessor()
    exs   = proc.get_test_examples(data, with_text_b=False,
                                   oov_method=args.oov_method, vocab=vocab)
    dl    = load_data(exs, label_list=None, output_mode='without_label',
                      tokenizer=tokenizer, max_seq_length=args.max_seq_length,
                      batch_size=args.eval_batch_size,
                      data_type=DataType.EVALUATION, model_type=args.model_type)
    logits=[]
    for batch in dl:
        batch=tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs={'input_ids':batch[0],'attention_mask':batch[1]}
            if args.model_type!='distilbert':
                inputs['token_type_ids']=batch[2] if args.model_type in ['bert','xlnet'] else None
            logits.append(model(**inputs)[0].cpu())
    logits=torch.cat(logits).softmax(dim=1)
    return torch.argmax(logits, dim=1).numpy()

def category_imputation(data_path: str, dataset_name: str, device, args):
    cat      = CategoryDataGenerator(data_path)
    tr_data  = cat.get_train_data(dataset_name, training_ratio=args.training_ratio)
    im_data  = cat.get_impute_data(dataset_name)
    table    = deepcopy(cat.data)
    train_times, imp_times = {}, {}

    for attr in tr_data:
        ix2v   = cat.ix_to_value[attr]
        labels = [str(i) for i in range(len(ix2v))]
        cfg_cls, mdl_cls, tok_cls = Config.MODEL_CLASSES[args.model_type]
        cfg      = cfg_cls.from_pretrained(args.model_name_or_path)
        cfg.num_labels = len(labels)
        tokenizer= tok_cls.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        model    = mdl_cls.from_pretrained(args.model_name_or_path, config=cfg) if args.pretrained=="pre" else mdl_cls(cfg)
        model.to(device)

        tr_df, vl_df = tr_data[attr]
        proc  = BERTProcessor()
        tr_exs = proc.get_train_examples(tr_df, with_text_b=False, oov_method=args.oov_method, vocab=tokenizer.get_vocab())
        vl_exs = proc.get_dev_examples(vl_df, with_text_b=False, oov_method=args.oov_method, vocab=tokenizer.get_vocab())

        t0 = time.process_time()
        best = train(model, tr_exs, vl_exs, tokenizer, device, labels, args)
        t1 = time.process_time()
        train_times[attr] = t1 - t0

        mis_df = im_data[attr]
        t2 = time.process_time()
        preds = impute_category(mis_df, best, tokenizer, device, args)
        t3 = time.process_time()
        imp_times[attr] = t3 - t2

        for i, (_, row) in enumerate(mis_df.iterrows()):
            if i < len(preds):
                v = ix2v.get(str(preds[i]), "")
                if v:
                    idx0 = table[table.id == row.id].index[0]
                    table.loc[idx0, attr] = v

    return table, train_times, imp_times

# ---- MAIN----
if __name__ == "__main__":
    args = read_arguments()
    device, n_gpu = initialize_gpu_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print("Using device:", device, "with", n_gpu, "GPUs")

    seed_everything(SEED)

    # load LaBSE once
    labse_tok   = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    labse_model = AutoModel.from_pretrained("sentence-transformers/LaBSE")
    labse_model.eval()
    labse_model.to(device)

    summary_rows = []

    for pct in PCT_NULLS:
        pattern = os.path.join(
            args.data_dir, f"*_{pct}_{SEED}_nonimputed_chunk*.csv"
        )
        all_files = sorted(glob.glob(pattern),
                           key=lambda fp: int(re.search(r'chunk(\d+)\.csv$', fp).group(1)))
        if not all_files:
            continue

        # group by dataset_name
        groups = {}
        for fp in all_files:
            ds = os.path.basename(fp).split("_")[0]
            groups.setdefault(ds, []).append(fp)

        for dataset_name, files in groups.items():
            if dataset_name not in ATTR_DICT:
                continue
            # load ground truth
            gt_fp = os.path.join(os.path.dirname(args.data_dir), f"{dataset_name}.csv")
            df_gt = pd.read_csv(gt_fp)

            # -------------------- ADDED: track cumulative start/end indices --------------------
            cumulative = 0

            for stage, chunk_path in enumerate(files):
                # compute start/end for this chunk
                df_chunk_len = len(pd.read_csv(chunk_path, dtype='str'))
                start_index = cumulative
                end_index   = cumulative + df_chunk_len - 1
                cumulative  = end_index + 1
                # -------------------------------------------------------------------------------

                # prepare input
                if stage == 0:
                    df_in = pd.read_csv(chunk_path, dtype='str')
                else:
                    prev_fp = os.path.join(
                        args.data_dir,
                        f"{dataset_name}_{stage}_{pct}_{SEED}_imputed_{stage-1}.csv"
                    )
                    df_prev = pd.read_csv(prev_fp, dtype='str')
                    df_new  = pd.read_csv(chunk_path, dtype='str')
                    df_in   = pd.concat([df_prev, df_new], ignore_index=True)

                # add id & save non-imputed
                df_in.insert(0, 'id', range(1, len(df_in)+1))
                base_full = os.path.basename(chunk_path).rsplit("_chunk",1)[0]
                inp_fp = os.path.join(args.data_dir, f"{base_full}_{stage}.csv")
                df_in.to_csv(inp_fp, index=False)

                # run imputation
                im_df, tr_times, imp_times = category_imputation(
                    inp_fp, dataset_name, device, args
                )

                # evaluate against ground truth
                inp_eval = pd.read_csv(inp_fp).drop(columns=['id'])
                imputed  = im_df.drop(columns=['id'], errors='ignore')
                mask     = inp_eval.isna() | (inp_eval == '')

                num_nulls = int(mask.values.sum())
                exact = 0
                preds, truths = [], []

                for col in inp_eval.columns:
                    for idx, is_null in mask[col].items():
                        if is_null:
                            pred_val = imputed.at[idx, col]
                            gt_val   = df_gt.at[idx, col]
                            if pd.notna(pred_val) and pd.notna(gt_val) and pred_val == gt_val:
                                exact += 1
                            preds.append(str(pred_val))
                            truths.append(str(gt_val))

                if preds:
                    enc_p = labse_tok(preds, padding=True, truncation=True, return_tensors='pt').to(device)
                    enc_t = labse_tok(truths, padding=True, truncation=True, return_tensors='pt').to(device)
                    with torch.no_grad():
                        out_p = labse_model(**enc_p)
                        out_t = labse_model(**enc_t)
                    emb_p = mean_pooling(out_p, enc_p['attention_mask'])
                    emb_t = mean_pooling(out_t, enc_t['attention_mask'])
                    sims  = F.cosine_similarity(emb_p, emb_t)
                    avg_sim = float(sims.mean().cpu())
                else:
                    avg_sim = 0.0

                # save imputed
                out_fp = os.path.join(
                    args.data_dir,
                    f"{dataset_name}_{stage+1}_{pct}_{SEED}_imputed_{stage}.csv"
                )
                im_df.drop(columns=['id'], errors='ignore').to_csv(out_fp, index=False)

                # record summary  (only additions: start_index & end_index)
                summary_rows.append({
                    "dataset":           dataset_name,
                    "pct_nulls":         pct,
                    "seed":              SEED,
                    "start_index":       start_index,   # ADDED
                    "end_index":         end_index,     # ADDED
                    "nulls":         num_nulls,
                    f"training_time_IPM_{training}":     sum(tr_times.values()),
                    f"imputation_time_IPM_{training}":   sum(imp_times.values()),
                    f"total_time_IPM_{training}":        sum(tr_times.values()) + sum(imp_times.values()),
                    f"exact_matches_IPM_{training}": exact,
                    f"avg_semantic_sim_IPM_{training}":  avg_sim
                })

    summary_df = pd.DataFrame(summary_rows)
    base_out_dir = f"/root/workspace/IPM/Data/{dataset}/{training}"
    filename     = f"IPM_evaluations_{training}_{dataset}_{SEED}.csv"
    summary_fp   = os.path.join(base_out_dir, filename)
    summary_df.to_csv(summary_fp, index=False)
    print("Saved summary to:", summary_fp)




