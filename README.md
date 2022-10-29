# EtriCA-storygeneration
This repository is the code and resources for the paper [EtriCA: Event-Triggered Context-Aware Story Generation Augmented by Cross Attention](https://arxiv.org/abs/2210.12463) 

To make sure everyone can easily reproduce our work, I will do my best to ensure every essential resource is included in this repository, and the README covers all the information you need to implement our work.

## Introduction
This project is implemented with **Pytorch**.

This project is implemented based on [pytorch-lightning](https://www.pytorchlightning.ai/) framework, a framework to ease the training of pytorch. If you dislike it, no worry, you can copy the model files (in `src/models`) and datasets files (in `src/modules`) to your own repository, and train them with your code.

All the pretrained model used here are downloaded from [Huggingface](https://huggingface.co/docs). E.g. [BART](https://aclanthology.org/2020.acl-main.703.pdf) is downloaded from [Hugginface: bart-base](https://huggingface.co/facebook/bart-base).

The code is organized as follows:
```markdown
├── datasets
   └── event-trigger		# expeirment group name
       ├── `roc-stories`        # a publicly available dataset: ROCStories
       ├── `writing-prompts`        # we use the processed dataset from HINT(from a paper)
   └── thu-coai-hint		# Testing HINT model will need it
├── preprocessing      # the code about automatical event extraction and event planning
├── resources      # resources for raw data, vanilla pretrained checkpoint, and so on.
├── src      # all the source code related to the models is put here
   └── configuration	# read the name, and you will know what it is for.
   └── models	
   └── modules	
   └── utils	
├── tasks      # the code to control experiments
   └── event-trigger 	# the code for training and testing
```
If there is something you could not understand, please try to find the explanation in [my paper]().

If you are a freshman in NLG area and feel hard to read the code, I prepared a story generation demo for you ([demo](https://github.com/tangg555/story-generation-demo)). 
I usually tried my best to design the data structure instead of writings code comments, because I believe a good code should be readable even without the code comments.

## Prerequisites
If you want to run this code, you have to at least satisfy the following requirement:
- Python 3 or Anaconda (mine is v3.8)
- [Pytorch](https://pytorch.org/) (mine is v1.11.0)
- transformers (a package for [huggingface](https://huggingface.co/facebook/bart-base)) v4.19.4
- [pytorch-lightning (a package)](https://www.pytorchlightning.ai/) v1.6.0
- all the packages listed in the file `requirements.txt` 

## Quick Start

### 1. Install packages
Install the aforementioned prerequisites, and run
```shell
python -r requirements.txt
```

### 2. Collect Datasets and Resources

`datasets` and `resources` are not included in the code, since their sizes are too large. 
Both of them can be downloaded from [datasets](https://www.dropbox.com/s/b007zce28ou52va/datasets.zip?dl=0)
and [resources](https://www.dropbox.com/s/wr9sxhhu4qteq2t/resources.zip?dl=0) . 
Unzip it at the base directory.

If you intend to preprocess the data by yourself, please read following instructions. Otherwise, please skip to the next section.

#### 2.1 Datasets

The **raw dataset** we suggest download

**Preprocess**

Put your downloaded raw dataset (we downloaded it from [HINT](https://github.com/thu-coai/HINT)) to `resources/raw_data`. 

Take roc-stories for an example, you need make it be `resources/raw_data/thu-coai-hint/roc-stories`.

Run `preprocessing/hint_roc_stories_helper.py`, and then `preprocessing/event_annotator.py`, and you will have `resources/datasets/event-plan/roc-stories`.

Similarly, if you want to run HINT as a story generation model for experiments, you need to download HINT dataset from [HINT](https://github.com/thu-coai/HINT), and make it to be `/datasets/thu-coai-hint/roc-stories`.

#### 2.2 Resources

The structure of resources should be like this:
```markdown
├── resources
   └── external-models		# put vanilla pretrained checkpoint
   └── raw_data		# for preprocessing
```
The huggingface pretrained models (e.g. `bart-base`) can be downloaded from [here](https://huggingface.co/facebook/bart-base). Or you can directly set `--model_name_or_path=facebook/bart-base`, the code will download it for you.

### 3. Run the code for training or testing

#### 3.1 Introduction

Please read the codes in `tasks`, and you will understand how it works.

**If you don't care the evaluation and experiments**, please only read following files:
- (1) `tasks/event-trigger/train.py` to train different models.
- (2) `tasks/event-trigger/test.py` to test different models.

#### 3.2  commands for EtriCA

In case you don't want to train **EtriCA** by yourself, some checkpoints are released for your convenience. 
- [event-lm-sbert-roc-stories](https://www.dropbox.com/s/uvbiwrm3ab2dgez/event-lm-sbert-roc-stories.tar.gz?dl=0)
- [event-lm-sbert-writing-prompts](https://www.dropbox.com/s/ehhox1hf6r24im7/event-lm-sbert-writing-prompts.tar.gz?dl=0)

Put it somewhere and restore it with a command. (referring to `python_commands.sh`)

The user parameters settings are located in 
`src/configuration/event_trigger/config_args.py`.

If you want to train **EtriCA** from scratch, please read following instructions.

**EtriCA** has two encoders, where one is for events, and the other is for leading contexts.

To simply the fine-tuning for the two different encoders, 
we firstly train `leading-plus-event-bart`, and then train 
`event-lm` (the ablated model for **EtriCA**) on the checkpoint of 
`leading-plus-event-bart`, and then train `event-lm-sbert` (**EtriCA**) on the 
checkpoint of `event-lm`.

Taking roc-stories dataset as an example.

For training:

(1) Train `leading-plus-event-bart`
```shell
python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=8e-5 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/event-trigger --model_name leading-plus-event-bart --experiment_name=leading-plus-event-bart-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0
```

(2) Train `event-lm`
```shell
python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=output/event-trigger/leading-plus-event-bart-roc-stories/best_tfmr \
 --output_dir=output/event-trigger --model_name event-lm --experiment_name=event-lm-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0
```

(3) Train `event-lm-sbert`
```shell
python tasks/event-trigger/train.py --data_dir=datasets/event-trigger/roc-stories\
 --learning_rate=1e-4 --train_batch_size=16 --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-lm-roc-stories/best_tfmr \
 --output_dir=output/event-trigger --model_name event-lm-sbert --experiment_name=event-lm-sbert-roc-stories\
 --val_check_interval=1.0 --limit_val_batches=10 --max_epochs=3 --accum_batches_args=4  --num_sanity_val_steps=0
```


For testing:
```shell
python tasks/event-trigger/test.py --data_dir=datasets/event-trigger/roc-stories\
  --eval_batch_size=10 --model_name_or_path=output/event-trigger/event-lm-sbert-roc-stories/best_tfmr \
  --output_dir=output/event-trigger --model_name event-lm-sbert --experiment_name=event-lm-sbert-roc-stories\
  --test_event_infix=_event
```

#### 3.3 Commands for experiments

We conduct a range of experiments to validate the effectiveness of our model, 
so it has plenty of commands. Please refer to the file `python_commands.sh` 
to select the command you want to execute.

## Notation
Some notes for this project.
### 1 - Additional directories and files in this project
```markdown
├── output  # this will be automatically created to put all the output stuff including checkpoints and generated text
├── .gitignore # used by git
├── requirement.txt # the checklist of essential python packages 
```
### 2 - Scripts for Downloading huggingface models
I wrote two scripts to download models from huggingface website.
One is `tasks/download_hf_models.sh`, and another is `src/utils/huggingface_helper.py`

## Citation
If you found this repository or paper is helpful to you, please cite our paper. It is accepted by EMNLP 2022 Findings, but currently the citations of EMNLP 2022 papers have not come out yet.

This is the arxiv citation:
```angular2
@article{tang2022etrica,
  title={EtriCA: Event-Triggered Context-Aware Story Generation Augmented by Cross Attention},
  author={Tang, Chen and Lin, Chenghua and Huang, Henglin and Guerin, Frank and Zhang, Zhihao},
  journal={arXiv preprint arXiv:2210.12463},
  year={2022}
}
```

