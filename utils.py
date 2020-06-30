from keras import backend as K
import numpy as np


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
 
def ignore_padding_accuracy(to_ignore=0):
    def ignore_pad_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_pad_accuracy
