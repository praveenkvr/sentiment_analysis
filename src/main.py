import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse
import pickle

MAX_LEN = 150

parser = argparse.ArgumentParser(description="Process inputs")
parser.add_argument('--text', help="pass text to analyze sentiment", required=True)

def get_tokenizer():
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def get_model():
    model = tf.keras.models.load_model('basic_attention_weights')
    model.summary()
    return model

if __name__ == '__main__':
    args = parser.parse_args()
    text = args.text
    print(f"TEXT ---- {text}")
    tokenizer = get_tokenizer()
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

    model = get_model()
    results = model.predict(padded_sequences)
    print(results)