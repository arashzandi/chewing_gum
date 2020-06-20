
import pickle

import argparse
from parseridge.corpus.treebank import Treebank
from keras.models import load_model

from utils import unzip, get_xy


def load(test_file):
    return Treebank(test_io=open(test_file), train_io=open(test_file))

def evaluate(test_file):
    treebank = load(test_file)
    with open('data.pkl', 'rb') as f:
        word2index, _, tag2index, maxlen = pickle.load(f)
    model = load_model('model.h5')
    test_sentences, test_tags = unzip(treebank.test_corpus)
    test_x, test_y = get_xy(test_sentences,
                            word2index,
                            test_tags,
                            tag2index,
                            maxlen)
    scores = model.evaluate(test_x, test_y)
    print(f"{model.metrics_names[1]}: {scores[1] * 100}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained tagger model.')
    parser.add_argument('test_file',
                        type=str,
                        help='Path to the test `.conllu` file')

    args = parser.parse_args()
    evaluate(args.test_file)