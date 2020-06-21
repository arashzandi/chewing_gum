import pickle

import numpy as np
import nltk
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, InputLayer, Bidirectional
from keras.layers import TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from parseridge.corpus.treebank import Treebank

from utils import unzip, get_unique_words, get_unique_tags
from utils import get_word2index, get_tag2index, get_index2tag
from utils import get_embedding_matrix


class DatasetInputOutput(object):
    def __init__(self, corpus, unknown_key='-OOV-'):
        self.input, self.output = unzip(corpus)
        self.unknown_key = unknown_key
        self.padding_length = None
        self.x = []
        self.y = []
    
    @staticmethod
    def pad(data, length):
        return pad_sequences(data, maxlen=length, padding='post')

    @staticmethod
    def to_categorical(sequences, categories):
        cat_sequences = []
        for s in sequences:
            cats = []
            for item in s:
                cats.append(np.zeros(categories))
                cats[-1][item] = 1.0
            cat_sequences.append(cats)
        return np.array(cat_sequences)

    def prepare_x(self, input_map):
        for s in self.input:
            s_int = []
            for w in s:
                try:
                    s_int.append(input_map[w.lower()])
                except KeyError:
                    s_int.append(input_map[self.unknown_key])
        
            self.x.append(s_int)
        self.x = self.pad(self.x, self.padding_length)

    def prepare_y(self, output_map):
        for s in self.output:
            self.y.append([output_map[t] for t in s])
        self.y = self.pad(self.y, self.padding_length)
        self.y = self.to_categorical(self.y, len(output_map))

    def prepare(self, input_map, output_map, padding_length):
        self.padding_length = padding_length
        self.prepare_x(input_map)
        self.prepare_y(output_map)
        

class Dataset(object):
    def __init__(self, treebank):
        self.treebank = treebank
        self._train = None
        self._validation = None
        self._test = None
        self.maxlen = None
    
    @property
    def train(self):
        if not self._train:
            self._train = DatasetInputOutput(self.treebank.train_corpus)
            self.maxlen = len(max(self._train.input, key=len))
        return self._train

    @property
    def validation(self):
        if not self._validation:
            self._validation = DatasetInputOutput(self.treebank.dev_corpus)
        return self._validation
    
    @property
    def test(self):
        if not self._test:
            self._test = DatasetInputOutput(self.treebank.test_corpus)
        return self._test


class Tagger(object):
    def __init__(self,
                 embeddings_length=100,
                 batch_size=128,
                 network_size=256,
                 epochs=30, 
                 init_from_file=False,
                 model_path='model.h5',
                 data_path='data.pkl'):
        self.model_path = model_path
        self.data_path = data_path
        self.embeddings_length = embeddings_length
        self.batch_size = batch_size
        self.network_size = network_size
        self.epochs = epochs
        self.dataset = None
        self._model = None
        self._maxlen = None
        self.word2index = {} 
        self.tag2index = {} 
        self.index2tag = {}
        if init_from_file:
            self.init_from_file()
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def init_from_file(self):
        with open(self.data_path, 'rb') as f:
            self.word2index, self.index2tag, self.tag2index, self._maxlen = pickle.load(f)
        self._model = load_model(self.model_path)

    def map(self):
        words = get_unique_words(self.dataset.train.input)
        tags = get_unique_tags(self.dataset.train.output)
        self.word2index = get_word2index(words)
        self.tag2index = get_tag2index(tags)
        self.index2tag = get_index2tag(tags)

    @property
    def model(self):
        if self._model:
            return self._model
        embedding_matrix = get_embedding_matrix(self.word2index)
        self._model = Sequential()
        self._model.add(InputLayer(input_shape=(self.dataset.maxlen, )))
        self._model.add(Embedding(len(self.word2index) + 1,
                            self.embeddings_length,
                            weights=[embedding_matrix],
                            input_length=self.dataset.maxlen,
                            trainable=True))
        self._model.add(Bidirectional(LSTM(self.network_size, return_sequences=True)))
        self._model.add(TimeDistributed(Dense(len(self.tag2index))))
        self._model.add(Activation('softmax'))
        self._model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=0.001),
                    metrics=['accuracy'])
        return self._model

    def save(self):
        with open(self.data_path, 'wb') as f:
            data = [self.word2index,
                    self.index2tag,
                    self.tag2index,
                    self.dataset.maxlen]
            pickle.dump(data, f)
        self.model.save(self.model_path)
        
    def train(self, train_file, validation_file):
        treebank = Treebank(train_io=open(train_file),
                            dev_io=open(validation_file))
        self.dataset = Dataset(treebank)
        self.map()

        self.dataset.train.prepare(self.word2index,
                                   self.tag2index,
                                   self.dataset.maxlen)
        self.dataset.validation.prepare(self.word2index,
                                        self.tag2index,
                                        self.dataset.maxlen)

        self.model # Initialize the model
        self.model.summary()
        self.model.fit(self.dataset.train.x,
                       self.dataset.train.y,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(self.dataset.validation.x,
                                        self.dataset.validation.y))

        self.save()
    
    def evaluate(self, test_file):
        treebank = Treebank(test_io=open(test_file),
                            train_io=open(test_file))
        self.dataset = Dataset(treebank)
        self.dataset.maxlen = self._maxlen
        self.dataset.test.prepare(self.word2index,
                                   self.tag2index,
                                   self.dataset.maxlen)
        self.model.evaluate(self.dataset.test.x, self.dataset.test.y)
    
    def generate(self, text_file):
        with open(text_file, 'r') as f:
            for sentence in f.readlines():
                print("****************************************")
                print('Text:')
                print(sentence)
                sentence = nltk.word_tokenize(sentence)
                tokenized = []
                for word in sentence:
                    try:
                        tokenized.append(self.word2index[word.lower()])
                    except KeyError:
                        tokenized.append(self.word2index['-OOV-'])
                tokenized = np.asarray([tokenized])
                padded = DatasetInputOutput.pad(tokenized, self._maxlen)
                prediction = self.model.predict(padded)
                for i, pred in enumerate(prediction[0][:len(sentence)]):
                    print(sentence[i], ' : ', self.index2tag[np.argmax(pred)])
        print("****************************************")