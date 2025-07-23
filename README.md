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
conda install -c conda-forge nvidia-apex
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```
