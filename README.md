# chewing_gum
Part-Of-Speech LSTM Tagger trained on Georgetown University Multilayer corpus.


See `THOUGHTS_LOG.md` for the transcription of the ideas and learnings.

## Setup
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en
```

## Train
```
python train.py UD_English-GUM/en_gum-ud-train.conllu UD_English-GUM/en_gum-ud-dev.conllu
```

## Evaluate
```
python evaluate.py UD_English-GUM/en_gum-ud-test.conllu
```

## Generate
```
python generate.py test_sentences.txt
```