# Andromeda

Andromeda serves as a natural surrogate of DBAs to answer a wide range of natural language (NL) questions on DBMS configuration issues, and to generate diagnostic suggestions to fix these issues. Nevertheless, directly prompting LLMs with these professional questions may result in overly generic and often unsatisfying answers. To this end, we propose a retrieval-augmented generation (RAG) strategy that effectively provides matched domain-specific contexts for the question from multiple sources. 

## Requirements

- python = 3.8.12
  
You can install multiple packages:

```
pip install -r requirements.txt
```

## 1. Quick start

### run example to generate hybrid-pipeline


```
python example.py
```

The file `example.py` is an example. Modify `query` according to your configuration.

- The **input** contains a user's NL question about DBSM configuration issue.
- The **output** is the troubleshooting configuration recommendation.

### You can find our code in `./core`.

## 2. Dataset

### You can find our benchmark in `./dataset` with link 

[https://1drv.ms/f/c/140409cb8fe0acca/Eo-VI-dPwVRNkald2yEdngIBJ0RZ616df-ZDtxXrITd2mg?e=WhrapQ ](https://1drv.ms/f/c/140409cb8fe0acca/Eo-VI-dPwVRNkald2yEdngIBLb-EM4i5vVVh_R718XQZRA?e=vFf6hd)

```python
-dataset
├── augment_train
│   └── mysql_forum_train_augment.json
│   └── mysql_so_train_augment.json
│   └── pg_so_train_augment.json
│   └── mysql_run_train_augment.json
├── historical_questions
│   └── mysql_forum_retrieval_data.json
│   └── mysql_so_retrieval_data.json
│   └── pg_so_retrieval_data.json
│   └── mysql_run_retrieval_data.json
├── manuals
│   └── mysql_manuals_data.json
│   └── mysql_manuals_data.json
├── test
│   └── mysql_forum_test_data.json
│   └── mysql_so_test_data.json
│   └── pg_so_test_data.json
│   └── mysql_run_test_data.json
├── train
│   └── mysql_forum_train_data.json
│   └── mysql_so_train_data.json
│   └── pg_so_train_data.json
│   └── mysql_run_train_data.json
-sbert_embeds
├── mysql_forum_retrieval_data.npy
├── mysql_forum_train_augment.npy
├── mysql_run_retrieval_data.npy
├── mysql_run_train_augment.npy
├── mysql_so_retrieval_data.npy
├── mysql_so_train_augment.npy
├── pg_so_retrieval_data.npy
├── pg_manuals_data.npy
├── mysql_manuals_data.npy
-sentence-transformers
├── all-mpnet-base-v2
```

The vectors in sbert_embeds are directly ebedded by model in sentence-transformers.

## 3. Results

You can find our generated prompts in `./results/prompt.json`.

You can find the results of LLM reasoning in `./reasoning_results`.

You can find the manual evaluation results in the Runnable setting in `./results/manual_evaluation_on_runnable_setting.json`.

## 4. Experiments

Please refer `./main.py` to see our experiment results.
