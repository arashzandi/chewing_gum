# chewing_gum
Part-Of-Speech LSTM Tagger trained on Georgetown University Multilayer corpus.


See `THOUGHTS_LOG.md` for the transcription of the ideas and learnings.

## Setup
A virtual environment for installing the python packages is required. Below 
Python `virtualenv` is used as an example.
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

The [GloVe Word Embeddings](https://nlp.stanford.edu/projects/glove/) are used
to enhance the peroformance of the model and can be downloaded and unpacked as
following:
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -d . glove.6B.zip glove.6B.100d.txt
rm glove.6B.zip
```

## Train POS Taggeer
To train the model on the GUM dataset you can use the following command. Please
note that the GUM dataset is added as a git submodule to this repository. The
training takes ca 10 minutes on the latest MacbookPro. After ca 20 epochs the 
networks shows symptoms of overfitting.
```
python train.py \
       UD_English-GUM/en_gum-ud-train.conllu \
       UD_English-GUM/en_gum-ud-dev.conllu
```

## Evaluate Model Performance on Test Data
To evaluate the trained model's performance on unseen data you can use this.
With the default hyperparameters a test accuracy of over 97.7% is acheived. 
```
python eval.py UD_English-GUM/en_gum-ud-test.conllu
```

## Generate POS Tags for Tokenized Sentences.
To see the model's inference you can add tokenized sentences to a text file 
and execute the following. To split the sentences into word tokens
[NLTK's pre-trained Punkt Tokenizer](https://www.nltk.org/_modules/nltk/tokenize/punkt.html)
is used. The POS-tagged words are printed out to stdout.
```
python generate.py <test_sentences.txt>
```