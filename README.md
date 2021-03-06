# PyNMT
<h2>Neural machine translation from English to German</h2>
NMT model is composed of one encoder for source language, one decoder for target language. 
However, the data required for
building such model requires language pair phrases. The following steps will explain NMT model.  
<h3>1. Preparing the Text Data</h3>
Before tokenization of data, it will be cleaned as below mentioned steps.

    Remove all non-printable characters.
    Remove all punctuation characters.
    Normalize all Unicode characters to ASCII (e.g. Latin characters).
    Normalize the case to lowercase.
    Remove any remaining tokens that are not alphabetic.
   

<h3>2. Encoder & Decoder</h3>
Once the data & vocabulary has been prepared, which is passed to the LSTM model for encoding a source language.
As well as another LSTM model for decoding encoded vectors into a target language.
  
Preprocessed data & trained models can be found [here](https://drive.google.com/drive/folders/1bSZtJAeMIVdhtxBQgSn7BgRzrF-05ZnX?usp=sharing)
### 3. Language model evaluation
Model evaluation can be done by generating a translated output sequence i.e the model can predict the entire output sequence.
Then the sequence of integers used to map the target langauge index-to-word dictionary to map back to words.

In addition, randomly selected each source phrase in a dataset and its corresponding predicted result will compared to 
the expected target phrase. As well as evaluation of BLEU scores will give a quantitative idea of model performance.

### Usage
##### Model training
```bash
python  PyNMT.py 
```
The above command starts training of language model with defaults parameters and saves the results to the corresponding 
directories. If you wish to tweak the parameters for training, below example give you an idea. 

```bash
python  PyNMT.py -b 32 -lr 0.001 -hs 200
```
##### Model evaluation
```bash
python  PyNMTevaluation.py 
```
### Available parameters:
```bash
usage: Implementation of neural language translator from English to German
       [-h] [-l LAYERS] [-b BATCH_SIZE] [-n NUM_EPOCHS] [-lr LR]
       [-hs HIDDEN_SIZE] [-d DROPOUT] [-bi]

optional arguments:
  -h, --help            show this help message and exit
  -l LAYERS, --layers LAYERS
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -n NUM_EPOCHS, --num_epochs NUM_EPOCHS
  -lr LR, --lr LR
  -hs HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
  -d DROPOUT, --dropout DROPOUT
  -bi, --bidirectional
```

### License
[MIT](https://github.com/Prabhu204/PyNMT/blob/master/LICENSE)
