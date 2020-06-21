
import argparse

from models import Tagger

def evaluate(test_file):
    tagger = Tagger(init_from_file=True)
    tagger.evaluate(test_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained tagger model.')
    parser.add_argument('test_file',
                        type=str,
                        help='Path to the test `.conllu` file')

    args = parser.parse_args()
    evaluate(args.test_file)