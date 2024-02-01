# **Requirements** <br />
python version 3.10 <br />
make environment <br />
```
pip install -r requirements.txt
```


# **Tuning** <br />
In order to do the tuning run: <br />
```
python tuning.<dataset>.py -o <optim> -m <model> (-lr or -all)
```
_dataset_ is one of the following (mnli, mrpc, sst2, stsb, cola) <br />
_optim_ is one of the following (adam, adamw, nadam, adamax, adabound, sgd, sgdm) <br />
choose -lr if you want to tune only learning rate or -all if you want to tune all hyperparameters <br />



# **Train** <br />
In order to train run: <br />
```
python train.<data>.py -o <optim> -m <model> -s <seed>
```
_dataset_ is one of the following (mnli, mrpc, sst2, stsb, cola) <br />
_optim_ is one of the following (adam, adamw, nadam, adamax, adabound, sgd, sgdm) <br />
_seed_ is the seed for splitting the dataset to train/validation/test <br />
