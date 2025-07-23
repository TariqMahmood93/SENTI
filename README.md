# SENtence Transformer-based data Imputation (SENT-I)

We propose a novel and innovative data imputation technique called Sentence Transformer-based Imputation (SENT-I).
 SENT-I leverages advanced indexing techniques to perform quick and accurate similarity searches, powered by 
state-of-the-art sentence transformer models. This approach is designed to enhance the accuracy and efficiency 
of dynamic data imputation tasks, providing a robust solution for handling missing data.


## Requirements

* Python 3.7.7
* PyTorch 1.9
* HuggingFace Transformers
* Spacy with the ``en_core_web_lg`` models
* NVIDIA Apex

```bash
# Define which dataset you want to run
dataset = "mammogram"
seed = 584
path = f"/root/workspace/Dynamic_Data_Imputation/v2/{dataset}"

config = {
    "adultsample": {"initial": 1000, "step": 100},
    "australian": {"initial": 230, "step": 23},
    "contraceptive": {"initial": 491, "step": 49},
    "credit": {"initial": 218, "step": 22},
    "flare": {"initial": 355, "step": 35},
    "imdb": {"initial": 1510, "step": 151},
    "mammogram": {"initial": 277, "step": 27},
    "thoracic": {"initial": 157, "step": 15},
}

initial = config[dataset]["initial"]
step = config[dataset]["step"]
%run /root/workspace/Dynamic_Data_Imputation/v2/zCode/main.py \
  --path {path} \
  --datasets {dataset} \
  --seeds {seed} \
  --cum_pcts 0.05 0.05 0.1 0.2 \
  --initial {initial} \
  --step {step} \
  --mode all
```
