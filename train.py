
import pickle

import argparse
from parseridge.corpus.treebank import Treebank

from utils import unzip, get_maps, get_xy, get_model


def load(train_file, dev_file):
    return Treebank(train_io=open(train_file), dev_io=open(dev_file))


def train(train_file, dev_file):
    treebank = load(train_file, dev_file)
    train_sentences, train_tags = unzip(treebank.train_corpus)
    val_sentences, val_tags = unzip(treebank.dev_corpus)
    maxlen = len(max(train_sentences, key=len))
    word2index, tag2index, index2tag = get_maps(train_sentences, train_tags)

    train_x, train_y = get_xy(train_sentences, word2index, train_tags, tag2index, maxlen)
    val_x, val_y = get_xy(val_sentences, word2index, val_tags, tag2index, maxlen)

    model = get_model(word2index, tag2index, maxlen)
    model.fit(train_x,
              train_y,
              batch_size=128,
              epochs=10,
              validation_data=(val_x, val_y))

    with open('data.pkl', 'wb') as f:
        pickle.dump([word2index, index2tag, tag2index, maxlen], f)
    model.save('model.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a POS Tagger.')
    parser.add_argument('train_file',
                        type=str,
                        help='Path to the training `.conllu` file')
    parser.add_argument('dev_file',
                        type=str,
                        help='Path to the dev `.conllu` file')

    args = parser.parse_args()
    train(args.train_file, args.dev_file)