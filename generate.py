
import pickle

import argparse
import numpy as np
from keras.models import load_model
import nltk

from utils import pad

nltk.download('punkt')


def generate(test_file):
    with open('data.pkl', 'rb') as f:
        word2index, index2tag, _, maxlen = pickle.load(f)
    model = load_model('model.h5')
    with open(test_file, 'r') as f:
        for sentence in f.readlines():
            print("****************************************")
            print('Text:')
            print(sentence)
            sentence = nltk.word_tokenize(sentence)
            tokenized_sentence = []
            for word in sentence:
                try:
                    tokenized_sentence.append(word2index[word.lower()])
                except KeyError:
                    tokenized_sentence.append(word2index['-OOV-'])
            tokenized_sentence = np.asarray([tokenized_sentence])
            padded_tokenized_sentence = pad(tokenized_sentence, maxlen)
            prediction = model.predict(padded_tokenized_sentence)
            for i, pred in enumerate(prediction[0][:len(sentence)]):
                print(sentence[i], ' : ', index2tag[np.argmax(pred)])
    print("****************************************")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate POS tags.')
    parser.add_argument('test_file',
                        type=str,
                        help='Path to the test `.txt` file')

    args = parser.parse_args()
    generate(args.test_file)