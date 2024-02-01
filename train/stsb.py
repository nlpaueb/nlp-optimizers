import argparse
from datasets import load_dataset, load_metric, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import random
import numpy as np
import torch
from transformers.file_utils import is_tf_available, is_torch_available
from sklearn.metrics import mean_squared_error

import os, sys
currDir = os.path.dirname(os.path.realpath(__file__))
rootDir = os.path.abspath(os.path.join(currDir, '..'))
if rootDir not in sys.path: # add parent dir to paths
    sys.path.append(rootDir)
    

parser = argparse.ArgumentParser(description='set model, optimizer and if you want to tune all hyperparams or only lr')

parser.add_argument("-o", "--optim", type=str, choices=['adabound','nadam','adamw','adam', 'adamax', 'sgd', 'sgdm'],
                    default = 'adam', help="choose optimizer")

parser.add_argument("-m", "--model", type=str, choices=['roberta', 'bert'],
                    default = 'bert', help="choose transformer model")

parser.add_argument("-s", "--seed", type=int, choices=[1,10,100,1000,10000],
                    default = 1, help="choose seed")




args = parser.parse_args()


if args.model == 'bert':
    model_checkpoint = 'distilbert-base-uncased'
else:
    model_checkpoint = 'distilroberta-base'
optim = args.optim

s=args.seed


task = "stsb"

if optim == 'nadam':
    from optimizers.Nadam import MyTrainingArguments , MyTrainer 
elif optim == "adabound":
    from optimizers.AdaBound import MyTrainingArguments , MyTrainer
elif optim == "adamax":
    from optimizers.AdaMax import MyTrainingArguments , MyTrainer
elif optim == "adamw":
    from optimizers.AdamW import MyTrainingArguments , MyTrainer
elif optim == "adam":
    from optimizers.Adam import MyTrainingArguments , MyTrainer
elif optim == "sgd":
    optim = 'sgdcustom'
    from optimizers.SGD import MyTrainingArguments , MyTrainer
elif optim == "sgdm":
    optim = 'sgdMcustom'
    from optimizers.SGDM import MyTrainingArguments , MyTrainer

    
#Custom seed

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
 
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf
 
        tf.random.set_seed(seed)
 
set_seed(1)




GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task]

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)



num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    return model






## Loading the dataset

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)
dataset1 = concatenate_datasets([dataset["train"],dataset["validation"]])
print (metric)

# Preprocessing the data

dataset2=dataset1.train_test_split(test_size=0.1666666666666,seed=s)

train = dataset2["train"]
valid = dataset2["test"].train_test_split(test_size=0.5,seed=s)["train"]
test = dataset2["test"].train_test_split(test_size=0.5,seed=s)["test"]

encoded_train = train.map(preprocess_function, batched=True)
encoded_valid= valid.map(preprocess_function, batched=True)
encoded_test = test.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[:, 0]
    results = metric.compute(predictions=predictions, references=labels)
    rmse = mean_squared_error(y_true=labels, y_pred=predictions, squared=False)
    pearson = round(results["pearson"], 2)
    spearmanr = round(results["spearmanr"], 2)
    return {
        'pearson': pearson, "spearmanr": spearmanr, 'rmse': rmse
    }

# Evaluate during training and a bit more often
# than the default to be able to prune bad trials early.
# Disabling tqdm is a matter of preference.
training_args = MyTrainingArguments( "1",do_eval=True, 
    eval_steps=500,
    optim = optim,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    logging_steps=500,
    save_total_limit = 2,
    warmup_steps= 500,
    num_train_epochs = 1,#12,
    load_best_model_at_end = True,
    logging_dir="1", 
    disable_tqdm=False
   )

trainer = MyTrainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_train,
    eval_dataset=encoded_valid,
    model_init=model_init,
    compute_metrics=compute_metrics
)

trainer.train()



f = open(task+'_'+args.model+'_'+ args.optim+'_seed_'+ str(s)+'.txt', 'w')

f.write('For train: ' + str(trainer.evaluate(encoded_train)) + '\n' + 'For valid: ' + str(
    trainer.evaluate(encoded_valid)) + '\n' + 'For test: ' + str(trainer.evaluate(encoded_test)))

f.close()