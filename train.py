import argparse

def train(train_file, dev_file):
    print(train_file, dev_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('train_file',
                        type=str,
                        help='Path to the training `conllu` file')
    parser.add_argument('dev_file',
                        type=str,
                        help='Path to the dev `conllu` file')

    args = parser.parse_args()
    train(args.train_file, args.dev_file)