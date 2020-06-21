# Chewing Gum
A Part-Of-Speech LSTM Tagger trained on Georgetown University Multilayer corpus.

## Why LSTM
Long Short Term Memory networks can remeber what happened before in a sequence 
and this memory does not fade out over time. A bi-directional LSTM can also 
make use of dependencies among words of a sentence from both sides (left and 
right). A little reading on the latest papers also confirmed my intuition.

See [THOUGHTS_LOG.md](THOUGHTS_LOG.md) for the transcription of the ideas and
learnings.

## Assumptions
The biggest assumption made is that the POS-Tagging is a solved problem. This
is can be largely wrong. Depending on the use-case other metrics than per-word
accuracy should be employed.
>With a per-word tagging accuracy of 97%, there is a probability of 45.6% that
a 20-word sentence (the average sentence length in the Brown corpus) contains
one or more tagging errors.
>
> -- [Is Part-of-Speech Tagging a Solved Task?](http://www.stefan-evert.de/PUB/GiesbrechtEvert2009_Tagging.pdf)

Another assumption is that all tags are equally important. For example, 
while punctutions or most frequent words can be easily tagged, tagging ambiguous
words is a harder task but they do not decrease the model performance due to 
their under-representation. Here again, based on the use-case, the metrics can
be changed to consider the class imbalance or the weights of the classes.

## Setup
A virtual environment for installing the python packages is required. Below 
Python `virtualenv` is used as an example.
```
git submodule update --init # to download the GUM dataset
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

## Further Potential Improvements (List of Trade-Offs)
 - Use a scheduler to tune hyper-parameters.
 - Re-Implement in pure Tensorflow or PyTorch for more maintainablity.
 - Add unit tests for the underlying functions and classes.
 - Monitor the number/percentage of the `unknown` words in the test data.

## Time Spent
I was only able to work on this project while either my son was at sleep or was
outside with my partner. A total of 12:30 hours were spent. Please see `git log`
or [THOUGHTS_LOG.md](THOUGHTS_LOG.md) for a more detailed protocol.

Friday 4pm - Email with code challenge was recieved.
Friday 9 - 11:30pm - I read the literature and implemented a prototype.
Saturday 12 am to 2pm - I spent some time on Pytorch Implementation.
Saturday 6pm to 10pm - I added `train`, `eval` and `generate` scripts.
Sunday 11am to 1pm - I refacored source code
Sunday 9pm to 11pm - I refinded the README and added comments.
Sunday 11pm - Email was sent back
