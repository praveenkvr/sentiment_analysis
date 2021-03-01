import os
import re
import tensorflow as tf
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from models import EMBEDDING_DIMS

tf.config.experimental.list_physical_devices('GPU')
nltk.download('wordnet')

stp_wrds = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

EMBEDDING_FILE = os.path.join(
    '..', 'dataset', 'glove', f'glove.6B.{EMBEDDING_DIMS}d.txt')


def clean_data(text):
    text = re.sub(r'@[^\s]+', '', text, re.UNICODE)  # remove @...
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))',
                  ' ', text, re.UNICODE)  # remove www
    # remove special characters
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
    text = re.sub(r'[0-9]+', '', text, re.UNICODE)  # remove numbers
    return text


def process_text(text):
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stp_wrds]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = [lemmatizer.lemmatize(word, 'v') for word in text]
    text = " ".join(text)
    return text


def get_glove_embeddings(file_name):
    embeddings_map = dict()
    with open(file_name,  encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddings = np.asarray(values[1:], dtype='float32')
            embeddings_map[word] = embeddings
    return embeddings_map


def build_embeddings(vocab_size, tokenizer):
    match_counter = 0
    embeddings = np.zeros((vocab_size, EMBEDDING_DIMS))
    emb_map = get_glove_embeddings(EMBEDDING_FILE)
    for word, i in tokenizer.word_index.items():
        if word in emb_map:
            embeddings[i] = emb_map.get(word)
            match_counter += 1
    return embeddings, match_counter
