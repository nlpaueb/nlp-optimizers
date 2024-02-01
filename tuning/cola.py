import argparse
import random
import numpy as np
import torch
from transformers.file_utils import is_tf_available, is_torch_available
from datasets import load_dataset, load_metric, concatenate_datasets
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
parser.add_argument('-lr', dest='only_lr', action='store_true',
                    help='Set the only_lr value to True.')
parser.add_argument('-all', dest='only_lr', action='store_false',
                    help='Set the only_lr value to False.')

args = parser.parse_args()


if args.model == 'bert':
    model_checkpoint = 'distilbert-base-uncased'
else:
    model_checkpoint = 'distilroberta-base'
optim = args.optim
only_lr = args.only_lr

task = "cola"

if optim == 'nadam':
    from optimizers.Nadam import MyTrainingArguments , MyTrainer, my_hp_space_optuna, my_hp_space_optuna_lr
elif optim == "adabound":
    from optimizers.AdaBound import MyTrainingArguments , MyTrainer, my_hp_space_optuna, my_hp_space_optuna_lr
elif optim == "adamax":
    from optimizers.AdaMax import MyTrainingArguments , MyTrainer, my_hp_space_optuna, my_hp_space_optuna_lr
elif optim == "adamw":
    from optimizers.AdamW import MyTrainingArguments , MyTrainer, my_hp_space_optuna, my_hp_space_optuna_lr
elif optim == "adam":
    from optimizers.Adam import MyTrainingArguments , MyTrainer, my_hp_space_optuna, my_hp_space_optuna_lr
elif optim == "sgd":
    optim = 'sgdcustom'
    from optimizers.SGD import MyTrainingArguments , MyTrainer, my_hp_space_optuna, my_hp_space_optuna_lr
elif optim == "sgdm":
    optim = 'sgdMcustom'
    from optimizers.SGDM import MyTrainingArguments , MyTrainer, my_hp_space_optuna, my_hp_space_optuna_lr
    
if only_lr:
    optuna = my_hp_space_optuna_lr
else:
    optuna = my_hp_space_optuna

# Custom seed
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



tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
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

num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    return model


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    print(id[0])
    if id[0] != trainer.run_id():
        global matthews
        matthews = []
    matthews.append(metric.compute(predictions=predictions, references=labels)['matthews_correlation'])
    print(metric.compute(predictions=predictions, references=labels)['matthews_correlation'])
    id[0] = trainer.run_id()
    return {
        'matthews_correlation': max(matthews),
    }

# Loading the dataset


actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

dataset1 = concatenate_datasets([dataset["train"], dataset["validation"]])



# SPLIT DATA (seed = 1)
s = 1
dataset2 = dataset1.train_test_split(test_size=0.1666666666666, seed=s, stratify_by_column='label')

train = dataset2["train"]
valid = dataset2["test"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')["train"]
test = dataset2["test"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')["test"]



encoded_train = train.map(preprocess_function, batched=True)
encoded_valid = valid.map(preprocess_function, batched=True)
encoded_test = test.map(preprocess_function, batched=True)

# Hyperparameter Search
matthews = [-3]
id = [-1]
# Evaluate during training and a bit more often
# than the default to be able to prune bad trials early.
# Disabling tqdm is a matter of preference.
training_args = MyTrainingArguments("1", do_eval=True,
                                    eval_steps=500,
                                    optim=optim,
                                    per_device_train_batch_size=4,
                                    per_device_eval_batch_size=4,
                                    evaluation_strategy="steps",
                                    logging_steps=500,
                                    save_total_limit=2,
                                    warmup_steps=500,
                                    num_train_epochs=10,
                                    load_best_model_at_end=True,
                                    logging_dir="1",
                                    disable_tqdm=False,
                                    )

trainer = MyTrainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_train,
    eval_dataset=encoded_valid,
    model_init=model_init,
    compute_metrics=compute_metrics
)

# Default objective is the sum of all metrics
# when metrics are provided, so we have to maximize it.
best_run1 = trainer.hyperparameter_search(
    direction="maximize",
    backend='optuna',
    n_trials=30,  # number of trials
    hp_space=optuna
)



# SPLIT DATA (seed = 10)
s = 10
dataset2 = dataset1.train_test_split(test_size=0.1666666666666, seed=s, stratify_by_column='label')

train = dataset2["train"]
valid = dataset2["test"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')["train"]
test = dataset2["test"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')["test"]



encoded_train = train.map(preprocess_function, batched=True)
encoded_valid = valid.map(preprocess_function, batched=True)
encoded_test = test.map(preprocess_function, batched=True)

# Hyperparameter Search
matthews = [-3]
id = [-1]
# Evaluate during training and a bit more often
# than the default to be able to prune bad trials early.
# Disabling tqdm is a matter of preference.
training_args = MyTrainingArguments("2", do_eval=True,
                                    eval_steps=500,
                                    optim=optim,
                                    per_device_train_batch_size=4,
                                    per_device_eval_batch_size=4,
                                    evaluation_strategy="steps",
                                    logging_steps=500,
                                    save_total_limit=2,
                                    warmup_steps=500,
                                    num_train_epochs=10,
                                    load_best_model_at_end=True,
                                    logging_dir="2",
                                    disable_tqdm=False,
                                    )

trainer = MyTrainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_train,
    eval_dataset=encoded_valid,
    model_init=model_init,
    compute_metrics=compute_metrics
)

# Default objective is the sum of all metrics
# when metrics are provided, so we have to maximize it.
best_run2 = trainer.hyperparameter_search(
    direction="maximize",
    backend='optuna',
    n_trials=30,  # number of trials
    hp_space=optuna
)


# SPLIT DATA (seed = 100)
s = 100
dataset2 = dataset1.train_test_split(test_size=0.1666666666666, seed=s, stratify_by_column='label')

train = dataset2["train"]
valid = dataset2["test"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')["train"]
test = dataset2["test"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')["test"]



encoded_train = train.map(preprocess_function, batched=True)
encoded_valid = valid.map(preprocess_function, batched=True)
encoded_test = test.map(preprocess_function, batched=True)

# Hyperparameter Search
matthews = [-3]
id = [-1]
# Evaluate during training and a bit more often
# than the default to be able to prune bad trials early.
# Disabling tqdm is a matter of preference.
training_args = MyTrainingArguments("3", do_eval=True,
                                    eval_steps=500,
                                    optim=optim,
                                    per_device_train_batch_size=4,
                                    per_device_eval_batch_size=4,
                                    evaluation_strategy="steps",
                                    logging_steps=500,
                                    save_total_limit=2,
                                    warmup_steps=500,
                                    num_train_epochs=10,
                                    load_best_model_at_end=True,
                                    logging_dir="3",
                                    disable_tqdm=False,
                                    )

trainer = MyTrainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_train,
    eval_dataset=encoded_valid,
    model_init=model_init,
    compute_metrics=compute_metrics
)

# Default objective is the sum of all metrics
# when metrics are provided, so we have to maximize it.
best_run3 = trainer.hyperparameter_search(
    direction="maximize",
    backend='optuna',
    n_trials=30,  # number of trials
    hp_space=optuna
)


# SPLIT DATA (seed = 1000)
s = 1000
dataset2 = dataset1.train_test_split(test_size=0.1666666666666, seed=s, stratify_by_column='label')

train = dataset2["train"]
valid = dataset2["test"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')["train"]
test = dataset2["test"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')["test"]



encoded_train = train.map(preprocess_function, batched=True)
encoded_valid = valid.map(preprocess_function, batched=True)
encoded_test = test.map(preprocess_function, batched=True)

# Hyperparameter Search
matthews = [-3]
id = [-1]
# Evaluate during training and a bit more often
# than the default to be able to prune bad trials early.
# Disabling tqdm is a matter of preference.
training_args = MyTrainingArguments("4", do_eval=True,
                                    eval_steps=500,
                                    optim=optim,
                                    per_device_train_batch_size=4,
                                    per_device_eval_batch_size=4,
                                    evaluation_strategy="steps",
                                    logging_steps=500,
                                    save_total_limit=2,
                                    warmup_steps=500,
                                    num_train_epochs=10,
                                    load_best_model_at_end=True,
                                    logging_dir="4",
                                    disable_tqdm=False,
                                    )

trainer = MyTrainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_train,
    eval_dataset=encoded_valid,
    model_init=model_init,
    compute_metrics=compute_metrics
)

# Default objective is the sum of all metrics
# when metrics are provided, so we have to maximize it.
best_run4 = trainer.hyperparameter_search(
    direction="maximize",
    backend='optuna',
    n_trials=30,  # number of trials
    hp_space=optuna
)


# SPLIT DATA (seed = 10000)
s = 10000
dataset2 = dataset1.train_test_split(test_size=0.1666666666666, seed=s, stratify_by_column='label')

train = dataset2["train"]
valid = dataset2["test"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')["train"]
test = dataset2["test"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')["test"]



encoded_train = train.map(preprocess_function, batched=True)
encoded_valid = valid.map(preprocess_function, batched=True)
encoded_test = test.map(preprocess_function, batched=True)

# Hyperparameter Search
matthews = [-3]
id = [-1]
# Evaluate during training and a bit more often
# than the default to be able to prune bad trials early.
# Disabling tqdm is a matter of preference.
training_args = MyTrainingArguments("5", do_eval=True,
                                    eval_steps=500,
                                    optim=optim,
                                    per_device_train_batch_size=4,
                                    per_device_eval_batch_size=4,
                                    evaluation_strategy="steps",
                                    logging_steps=500,
                                    save_total_limit=2,
                                    warmup_steps=500,
                                    num_train_epochs=10,
                                    load_best_model_at_end=True,
                                    logging_dir="5",
                                    disable_tqdm=False,
                                    )

trainer = MyTrainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_train,
    eval_dataset=encoded_valid,
    model_init=model_init,
    compute_metrics=compute_metrics
)

# Default objective is the sum of all metrics
# when metrics are provided, so we have to maximize it.
best_run5 = trainer.hyperparameter_search(
    direction="maximize",
    backend='optuna',
    n_trials=30,  # number of trials
    hp_space=optuna
)

if only_lr:
    params = 'only lr'
else:
    params = 'all hyperparameters'

f = open(task + '_'+ args.model + '_' + args.optim +'_bestruns.txt', 'w')

f.write("for seed: "+str(1) + '\n')
f.write(str(best_run1) + '\n')
f.write("for seed: "+str(10) + '\n')
f.write(str(best_run2) + '\n')
f.write("for seed: "+str(100) + '\n')
f.write(str(best_run3) + '\n')
f.write("for seed: "+str(1000) + '\n')
f.write(str(best_run4) + '\n')
f.write("for seed: "+str(10000) + '\n')
f.write(str(best_run5) + '\n')
f.write( params + " tuned")
f.close()