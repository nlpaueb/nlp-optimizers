import argparse
import random
import numpy as np
import torch
from transformers.file_utils import is_tf_available, is_torch_available
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, load_metric, concatenate_datasets
from sklearn.metrics import f1_score,auc, precision_recall_curve


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

task = "mnli"

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

actual_task = "mnli" if task == "mnli-mm" else task
# Loading Dataset
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)

num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    return model




def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = metric.compute(predictions=predictions, references=labels)['accuracy']

    f1_0 = f1_score(y_true=labels, y_pred=predictions, labels=[0], average=None)
    f1_1 = f1_score(y_true=labels, y_pred=predictions, labels=[1], average=None)
    f1_2 = f1_score(y_true=labels, y_pred=predictions, labels=[2], average=None)
    f1 = f1_score(y_true=labels, y_pred=predictions, average=None)

    f1_macro = f1_score(y_true=labels, y_pred=predictions, average='macro')

    return {
        'accuracy': acc,
        'macro_f1': f1_macro,
        'f1_0': f1_0[0],
        'f1_1': f1_1[0],
        'f1_2': f1_2[0]

    }




#AUC


def auc_multiclass(actual_class, pred_class):
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    auc_dict = {}
    prec = []
    rec = []
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [x[per_class] for x in pred_class]
        # new_pred_class = [pred_class[x][per_class] if np.argmax(x) in other_class else max(x) for x in pred_class]

        # using the sklearn metrics method to calculate the auc
        precision, recall, thresholds = precision_recall_curve(new_actual_class, new_pred_class, pos_label=1)
        auc1 = auc(recall, precision)
        prec.append(precision)
        rec.append(recall)
        print("For class: " + str(per_class))
        print("Precision " + str(per_class) + " is: " + str(prec[per_class]))
        print("Recall " + str(per_class) + " is: " + str(rec[per_class]))
        #  print (auc(recall,precision))
        #  print (unique_class)
        #  print (other_class)
        #  print (new_actual_class)
        #  print (new_pred_class)

        auc_dict[per_class] = auc1

    return auc_dict, prec, rec



a = dataset["validation_matched"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')['train']
b = dataset["validation_mismatched"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')['train']
c = dataset["validation_matched"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')['test']
d = dataset["validation_mismatched"].train_test_split(test_size=0.5, seed=s, stratify_by_column='label')['test']

valid = concatenate_datasets([a, b])
test = concatenate_datasets([c, d])
train = \
dataset["train"].train_test_split(test_size=1 - 50000 / len(dataset["train"]), seed=s, stratify_by_column='label')[
    'train']

encoded_train = train.map(preprocess_function, batched=True)
encoded_valid = valid.map(preprocess_function, batched=True)
encoded_test = test.map(preprocess_function, batched=True)

# Hyperparameter Space


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
                                    num_train_epochs=1,
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

trainer.train()



a1 = trainer.predict(encoded_train)
a2 = trainer.predict(encoded_valid)
a3 = trainer.predict(encoded_test)

b = trainer.evaluate(encoded_train)
c = trainer.evaluate(encoded_valid)
d = trainer.evaluate(encoded_test)

auc_dict1, prec1, rec1 = auc_multiclass(np.array(train['label']), a1.predictions)
auc_dict2, prec2, rec2 = auc_multiclass(np.array(valid['label']), a2.predictions)
auc_dict3, prec3, rec3 = auc_multiclass(np.array(test['label']), a3.predictions)

f = open(task+'_'+args.model+'_'+ args.optim+'_seed_'+ str(s)+'.txt', 'w')

f.write('TRAIN: \n')
f.write('Accuracy: ' + str(b.get('eval_accuracy')) + '\n')
f.write('F1_0: ' + str(b.get('eval_f1_0')) + '\n')
f.write('F1_1: ' + str(b.get('eval_f1_1')) + '\n')
f.write('F1_2: ' + str(b.get('eval_f1_2')) + '\n')
f.write('macro_f1: ' + str(b.get('eval_macro_f1')) + '\n')
f.write('AUC_0: ' + str(auc_dict1[0]) + '\n')
f.write('AUC_1: ' + str(auc_dict1[1]) + '\n')
f.write('AUC_2: ' + str(auc_dict1[2]) + '\n')
f.write('MACRO AUC: ' + str((auc_dict1[2] + auc_dict1[0] + auc_dict1[1]) / 3) + '\n')
f.write('\n')

f.write('VALID: \n')
f.write('Accuracy: ' + str(c.get('eval_accuracy')) + '\n')
f.write('F1_0: ' + str(c.get('eval_f1_0')) + '\n')
f.write('F1_1: ' + str(c.get('eval_f1_1')) + '\n')
f.write('F1_2: ' + str(c.get('eval_f1_2')) + '\n')
f.write('macro_f1: ' + str(c.get('eval_macro_f1')) + '\n')
f.write('AUC_0: ' + str(auc_dict2[0]) + '\n')
f.write('AUC_1: ' + str(auc_dict2[1]) + '\n')
f.write('AUC_2: ' + str(auc_dict2[2]) + '\n')
f.write('MACRO AUC: ' + str((auc_dict2[2] + auc_dict2[0] + auc_dict2[1]) / 3) + '\n')
f.write('\n')

f.write('TEST: \n')
f.write('Accuracy: ' + str(d.get('eval_accuracy')) + '\n')
f.write('F1_0: ' + str(d.get('eval_f1_0')) + '\n')
f.write('F1_1: ' + str(d.get('eval_f1_1')) + '\n')
f.write('F1_2: ' + str(d.get('eval_f1_2')) + '\n')
f.write('macro_f1: ' + str(d.get('eval_macro_f1')) + '\n')
f.write('AUC_0: ' + str(auc_dict3[0]) + '\n')
f.write('AUC_1: ' + str(auc_dict3[1]) + '\n')
f.write('AUC_2: ' + str(auc_dict3[2]) + '\n')
f.write('MACRO AUC: ' + str((auc_dict3[2] + auc_dict3[0] + auc_dict3[1]) / 3) + '\n')
f.write('\n')

f.close()