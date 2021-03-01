import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from twitter_api import process_tweets

MAX_LEN = 160

tokenizer = None
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)
    print("Tokenizer loaded .....")

model_filename = "basic_attention_weights4"
model = tf.keras.models.load_model(model_filename)
print("Model loaded")


def analyze_texts(texts):
    texts = process_tweets(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")
    results = model.predict(padded_sequences)
    results = results.tolist()
    return results
