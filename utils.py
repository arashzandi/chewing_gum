import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam

def unzip(corpus):
    return split(tag_sentences(corpus))

def split(tagged_sentences):
    sentences, sentence_tags =[], [] 
    for tagged_sentence in tagged_sentences:
        sentence, tags = zip(*tagged_sentence)
        sentences.append(np.array(sentence))
        sentence_tags.append(np.array(tags))
    return sentences, sentence_tags


def tag_sentences(corpus):
    for sentence in corpus.sentences:
        result = []
        for token in sentence.tokens:
            result.append((token.form, token.upostag))
        yield result

def get_unique_words(train_sentences):
    words = set()
    for s in train_sentences:
        for w in s:
            words.add(w.lower())
    return words

def get_unique_tags(train_tags):
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

def get_maps(sentences, tags):
    words = get_unique_words(sentences)
    tags = get_unique_tags(tags)
    word2index = get_word2index(words)
    tag2index = get_tag2index(tags)
    index2tag = get_index2tag(tags)
    return word2index, tag2index, index2tag

def pad(data, length):
    return pad_sequences(data, maxlen=length, padding='post')

def get_x(sentences, word2index, maxlen):
    x = []
    for s in sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
    
        x.append(s_int)
    return pad_sequences(x, maxlen=maxlen, padding='post')

def get_y(tags, tag2index, maxlen):
    y = []
    for s in tags:
        y.append([tag2index[t] for t in s])
    return pad_sequences(y, maxlen=maxlen, padding='post')

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


def get_xy(sentences, word2index, tags, tag2index, maxlen):
    x = get_x(sentences, word2index, maxlen)
    y = to_categorical(get_y(tags, tag2index, maxlen), len(tag2index))
    return x, y

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
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_model(word2index, tag2index, maxlen):
    embedding_matrix = get_embedding_matrix(word2index)
    model = Sequential()
    model.add(InputLayer(input_shape=(maxlen, )))
    model.add(Embedding(len(word2index) + 1,
                        100,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=True))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(tag2index))))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=0.001),
                metrics=['accuracy'])
    
    model.summary()
    return model