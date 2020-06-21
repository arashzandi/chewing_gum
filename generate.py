import argparse

from models import Tagger

def generate(text_file):
    tagger = Tagger(init_from_file=True)
    tagger.generate(text_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate POS tags.')
    parser.add_argument('text_file',
                        type=str,
                        help='Path to the test `.txt` file')

    args = parser.parse_args()
    generate(args.text_file)