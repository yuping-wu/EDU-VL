# EDU-VL
Source code for paper "EDU-level Extractive Summarization with Varying Summary Lengths"


## Dependencies
- Python 3.7.11
- PyTorch 1.8.0
- pyrouge 0.1.3
- transformers 4.12.5
- allennlp 2.8.0
- [pythonrouge] (https://github.com/tagucci/pythonrouge)
  - Please run command ```pytouge_set_rouge_path``` to setup the ROUGE package.
- For more requirements, please check `requirements.txt`

## Pre-process
- Data preparation: we adapt pre-processing steps from [DiscoBERT] (https://github.com/jiacheng-xu/DiscoBERT) to pre-process data.

## Train
Setup model configuration at configuration file `config/model.json`. Run the following command to train model:

```
python main.py
```

Output:
- `xxxx/model.tar.gz`: The trained model would be automatically saved under a folder with randomly generated name.
- `model_info.txt`: #params of model and model architecture

## Test
Run the following command to test model:

```
allennlp evaluate PATH-TO-model.tar.gz PATH-TO-test-dataset --output-file evaluation.txt --cuda-device 0 --include-package model
```
Output:
- `evaluation.txt`: testing result
