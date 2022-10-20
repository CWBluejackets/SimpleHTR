import os

# Filenames and paths to data.

# default value
model_dir = '../model'

fn_corpus = '../data/corpus.txt'


def set_model_dir(model_dir0):
    global model_dir
    model_dir = model_dir0
    return


def get_model_dir():
    return model_dir


def get_fn_char_list():
    return os.path.join(model_dir, 'charList.txt')


def get_fn_summary():
    return os.path.join(model_dir, 'summary.json')
