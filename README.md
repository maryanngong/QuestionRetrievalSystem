# Question Retrieval Project
##### Authors: Maryann Gong (mmgong), Nick Matthews (nickjm)

## Description
This github repo implements the final project for MIT's 6.864 Advanced Natural Language Processing, as taught in
Fall 2017 by Professor Regina Barzilay. You can read the full description and results in `project_report.pdf`.

## Project Structure
- `Android` Android Corpus Data (git submodule)
- `AskUbuntu` Ubuntu Corpus Data (git submodule)
- `code` All project code
    - `data` Includes data processing utils in `myio.py`, embeddings, and temporary data storage
    - `models`: All models are defined in `model_utils.py`
    - `train`: `train_utils.py` includes the training procedure and helper functions such as the evaluation procedure.
    - `scripts`: Includes the main entry point `main.py`
- `requirements.txt` requirements to run this code on a MacOS machine
- `requirements_linux.txt` requirements to run this code on a linux machine with CUDA 8


## Training the Models

### Prerequisites
Please run:
```
git submodule init
git submodule update
```
If you are going to run models in the domain adaptation setting, please download lowercased GloVe embeddings to the `code/data`
directory and run:
```
cd code/data
python myio.py
```
To safely run our models, and ensure they have access to everything in the data directory please `cd` to `code/data`.
Then you can run the program via:
```
python ../scripts/main.py
```
To train the model you just need to add the train flag:
```
python ../scripts/main.py --train
```
To see all the possible flags you can set, run
```
python ../scripts/main.py --help
```

### Saving and Loading
Specify a path to save your model and the model details should be serialized and appended to your specified path. This
allows you to run multiple configurations of hyper-parameters with the same save path.
```
python ../scripts/main.py --train --save_path='your-model-path-here'
```
Loading from a saved model is just as easy. Be sure to specify the full path of the saved file, including what was
automatically appended to the model path you specified in the previous step
```
python ../scripts/main.py --train --snapshot='full-model-path-here'
```

### Running with CUDA
Simply add the cuda flag. For example:
```
python ../scripts/main.py --train --cuda
```
Or you can use the shortcut `-c`. You probably also want to specify a specific gpu to run on by adding `CUDA_VISIBLE_DEVICES=0` before the rest of your command. Replace zero with the desired GPU ID

### Part 1 - In Domain Models
To train a CNN encoder model with a successful model architecture and configuration of hyperparameters try:
```
python ../scripts/main.py -t --epochs=20 --batch_size=32 --lr=0.001 --model_name='cnn3' --num_hidden=512 --margin=0.5
```

### Part 2 - Domain Adaptation
**TF-IDF Baseline:** To run tfidf on the entire android corpus
```
python ../scripts/main.py -a --model_name='tfidf'
```
**Direct Transfer Baseline:** To train a CNN encoder model as in Part 1, but evaluate performance on the Android corpus with AUC_0.05:
```
python ../scripts/main.py -at --epochs=20 --batch_size=32 --lr=0.001 --model_name='cnn3' --num_hidden=512 --margin=0.2
```
**Domain Adaptation:** Just add the `-d` flag, and optionally specify hyper-parameters for the discriminator model. For example,
```
python ../scripts/main.py -adt --epochs=20 --batch_size=32 --lr=0.001 --model_name='cnn3' --num_hidden=512 --margin=0.2 --lr2=-0.001 --lam=0.001 --dropout=0.2
```
**Reconstruction Loss:** TODO
... need to merge in from separate branch `exploration_autoencoder`
```
CODE HERE
```
**WGAN:** To train using the WGAN approach add the `-g` flag (for GAN setup) and `-w` flag to change loss, parameter clamping, and model outputs for following the original WGAN paper. You can add some other flags to specify the exact configuration we used to get our best results:
```
python ../scripts/main.py -adtwg --epochs=20 --batch_size=32 --lr=0.001 --num_hidden=512 --margin=0.2 --dropout=0.2 --dropout_d=0.1 --dropout_t=0.1 --complex_transformer --simple_discriminator --pad_max
```
Note: The encoder model can't be set when using WGAN, and we also fix the learning rates to a stable value for WGAN. Please use the `--help` flag to see other flags that may be relevant for you.
