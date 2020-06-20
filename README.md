# chewing_gum
Part-Of-Speech LSTM Tagger trained on Georgetown University Multilayer corpus.


See `THOUGHTS_LOG.md` for the transcription of the ideas and learnings.

## Setup
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -d . glove.6B.zip glove.6B.100d.txt
```

## Train
```
python train.py UD_English-GUM/en_gum-ud-train.conllu UD_English-GUM/en_gum-ud-dev.conllu
```

