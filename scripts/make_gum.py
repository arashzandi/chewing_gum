# Based on https://nlpforhackers.io/lstm-pos-tagger-keras/

import pickle
import numpy as np
from parseridge.corpus.treebank import Treebank
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from keras.models import load_model

def get_embedding_matrix(word2index):
    embeddings_index = {}
    with open('glove.6B.100d.txt', encoding="utf8") as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_matrix = np.random.random((len(word2index) + 1, 100))
    for word, i in word2index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embeddings_index will remain unchanged and thus will be random.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def tag_sentences(corpus):
    for sentence in corpus.sentences:
        result = []
        for token in sentence.tokens:
            result.append((token.form, token.upostag))
        yield result
 
def split(tagged_sentences):
    sentences, sentence_tags =[], [] 
    for tagged_sentence in tagged_sentences:
        sentence, tags = zip(*tagged_sentence)
        sentences.append(np.array(sentence))
        sentence_tags.append(np.array(tags))
    return sentences, sentence_tags

def load_data():
    return Treebank(
        train_io=open("../UD_English-GUM/en_gum-ud-train.conllu"),
        dev_io=open("../UD_English-GUM/en_gum-ud-dev.conllu"),
        test_io=open("../UD_English-GUM/en_gum-ud-test.conllu"))

def get_words(train_sentences):
    words = set()
    for s in train_sentences:
        for w in s:
            words.add(w.lower())
    return words

def get_tags(train_tags):
    tags = set()
    for ts in train_tags:
        for t in ts:
            tags.add(t)
    return tags

def get_word2index(words):
    word2index = {w: i + 2 for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0  # The special value used for padding
    word2index['-OOV-'] = 1  # The special value used for OOVs
    return word2index

def get_tag2index(tags):
    tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
    tag2index['-PAD-'] = 0  # The special value used to padding
    return tag2index

def get_index2tag(tags):
    index2tag = {}
    for i, tag in enumerate(tags):
        index2tag[i+1] = tag
    return index2tag

def get_x(sentences, word2index):
    x = []
    for s in sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
    
        x.append(s_int)
    return x

def get_y(tags, tag2index):
    y = []
    for s in tags:
        y.append([tag2index[t] for t in s])
    return y

def get_model(word2index, tag2index, embedding_matrix):
    model = Sequential()
    model.add(InputLayer(input_shape=(MAX_LENGTH, )))
    model.add(Embedding(len(word2index) + 1,
                        100,
                        weights=[embedding_matrix],
                        input_length=MAX_LENGTH,
                        trainable=True))
    # model.add(Embedding(len(word2index), 128))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(tag2index))))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=0.001),
                metrics=['accuracy'])
    
    model.summary()
    return model

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

treebank = load_data()
train_sentences, train_tags = split(tag_sentences(treebank.train_corpus))
validation_sentences, validation_tags = split(tag_sentences(treebank.dev_corpus))
test_sentences, test_tags = split(tag_sentences(treebank.test_corpus))

words = get_words(train_sentences)
tags = get_tags(train_tags)
word2index = get_word2index(words)
tag2index = get_tag2index(tags)
index2tag = get_index2tag(tags)
train_sentences_X = get_x(train_sentences, word2index)
test_sentences_X = get_x(test_sentences, word2index)
validation_sentences_X = get_x(validation_sentences, word2index)
train_tags_y = get_y(train_tags, tag2index)
test_tags_y = get_y(test_tags, tag2index)
validation_tags_y = get_y(validation_tags, tag2index)

MAX_LENGTH = len(max(train_sentences_X, key=len))
print(MAX_LENGTH)  # 271

def pad(data, length):
    return pad_sequences(data, maxlen=length, padding='post')

train_sentences_X = pad(train_sentences_X, MAX_LENGTH)
test_sentences_X = pad(test_sentences_X, MAX_LENGTH)
validation_sentences_X = pad(validation_sentences_X, MAX_LENGTH)
train_tags_y = pad(train_tags_y, MAX_LENGTH)
test_tags_y = pad(test_tags_y, MAX_LENGTH)
validation_tags_y = pad(validation_tags_y, MAX_LENGTH)

embedding_matrix = get_embedding_matrix(word2index)
model = get_model(word2index, tag2index, embedding_matrix)

cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
cat_test_tags_y = to_categorical(test_tags_y, len(tag2index))
cat_validation_tags_y = to_categorical(validation_tags_y, len(tag2index))

try:
    with open('data.pkl', 'rb') as f:
        word2index, index2tag = pickle.load(f)
    model = load_model('model.h5')
except:
    model.fit(train_sentences_X,
            cat_train_tags_y,
            batch_size=128,
            epochs=10,
            validation_data=(validation_sentences_X, cat_validation_tags_y))
    scores = model.evaluate(test_sentences_X, cat_test_tags_y)
    print(f"{model.metrics_names[1]}: {scores[1] * 100}")   # acc: 97.63
    with open('data.pkl', 'wb') as f:
        pickle.dump([word2index, index2tag], f)
    model.save('model.h5')

sentence = 'America likes Canada every night at 9'
sentence = ['*root*'] + sentence.split(' ')
tokenized_sentence = []
for word in sentence:
    try:
        tokenized_sentence.append(word2index[word.lower()])
    except KeyError:
        tokenized_sentence.append(word2index['-OOV-'])

tokenized_sentence = np.asarray([tokenized_sentence])
padded_tokenized_sentence = pad(tokenized_sentence, MAX_LENGTH)
prediction = model.predict(padded_tokenized_sentence)
for i, pred in enumerate(prediction[0][:len(sentence)]):
	print(sentence[i], ' : ', index2tag[np.argmax(pred)])

