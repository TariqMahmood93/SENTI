# Semantic-Aware Data Imputation in Dynamic Relational Databases via Pre-trained Language Models

We propose a novel and innovative data imputation technique called Sentence Transformer-based Imputation (SENT-I) to address the challenges of Dynamic Data Imputation (DDI)—the task of imputing missing values in incrementally updated databases. SENT-I leverages advanced indexing techniques to perform quick and accurate similarity searches by exploiting the inference capabilities of state-of-the-art sentence transformer models, without requiring any finetuning. This approach is designed to improve both the accuracy and efficiency, offering a robust and scalable solution for handling missing data over time.


## Requirements

* Python 3.7.7
* PyTorch 1.9
* HuggingFace Transformers
* Spacy with the ``en_core_web_lg`` models
* NVIDIA Apex
* 
## To run the SENTI algorithm
____________________________________________________________________
```bash
dataset = "adultsample"
seed = 1234
path = f"/root/workspace/SENTI/{dataset}"

config = {
    "adultsample": {"initial": 1000, "step": 100},
    "australian": {"initial": 230, "step": 23},
    "contraceptive": {"initial": 491, "step": 49},
    "credit": {"initial": 218, "step": 22},
    "imdb": {"initial": 1510, "step": 151},
}

initial = config[dataset]["initial"]
step = config[dataset]["step"]

%run /root/workspace/SENTI/Code/main.py \
  --path {path} \
  --datasets {dataset} \
  --seeds {seed} \
  --cum_pcts 0.05 0.05 0.1 0.2 \
  --initial {initial} \
  --step {step} \
  --mode all
```
**Arguments**

- `--path`  Path to the working directory (where input files are read from and outputs/logs are written).

- `--datasets`  One or more dataset identifiers to process (e.g., `adultsample`, `australian`). Space‑separate or repeat the flag, depending on the parser.

- `--seeds`  Random seed(s) for reproducibility of injection/imputation routines.

- `--cum_pcts`  Cumulative percentages of nulls to inject (`5`,`10`,`20`,`40`) in successive rounds.

- `--initial` Starting chunk size.

- `--step`  Incrementally added new tuples.

- `--mode`  
  Controls what the script does:  
  - `inject`   – only inject nulls  
  - `SENT-I` – only run imputation (expects `*_nonimputed.csv` already present)  
  - `all`     – perform injection **then** imputation in sequence

## To run the IPM algorithm

**To run the IPM algorithm, check the README file in the IPM directory.**
____________________________________________________________________
```bash
dataset  = "adultsample"
seed     = 1234
training = "fixed" # "fixed" or "Retraining"
path      = f"/root/workspace/IPM-main/data/{dataset}/{training}"
code_file = f"/root/workspace/IPM-main/src/ipm_multi_{training}.py"

%run $code_file \
  --model_type roberta \
  --model_name_or_path deepset/roberta-base-squad2 \
  --data_dir $path \
  --dataset $dataset \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --max_seq_length 75 \
  --num_epochs 2 \
  --seed $seed

```
