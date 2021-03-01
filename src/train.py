import os
import re
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from util import clean_data, process_text, build_embeddings
from models import build_lstm_with_attention, build_multi_head_attention, EMBEDDING_DIMS


VOCAB_SIZE = 300000
MAX_LEN = 160
BUFFER_SIZE = 512
BATCH_SIZE = 512
EPOCHS = 25


def load_data():
    data_path = os.path.join('..', 'dataset', 'data.csv')
    data = pd.read_csv(data_path, names=[
        'label', 'time', 'date', 'query', 'username', 'text'])
    data = data[['label', 'text']]
    return data


def process_data(data):
    data.loc[(data['label'] == 4), 'label'] = 1
    data = data[data['text'].apply(lambda x: isinstance(x, str))]
    data['text'] = data['text'].apply(clean_data)
    data['text'] = data['text'].apply(process_text)

    return data


# load and process data
data = load_data()
data = process_data(data)

# split data
train, test = train_test_split(data, test_size=0.1)
test, val = train_test_split(test, test_size=0.5)

if os.path.exists('tokenizer.pickle'):
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
        print('Tokenizer loaded .....')
else:
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<oov>')
    tokenizer.fit_on_texts(train['text'])

train_seq = tokenizer.texts_to_sequences(train['text'])
X_train = pad_sequences(train_seq, maxlen=MAX_LEN, padding='post')

test_seq = tokenizer.texts_to_sequences(test['text'])
X_test = pad_sequences(test_seq, maxlen=MAX_LEN, padding='post')

val_seq = tokenizer.texts_to_sequences(val['text'])
X_val = pad_sequences(val_seq, maxlen=MAX_LEN, padding='post')

# save tokenizer
if not os.path.exists('tokenizer.pickle'):
    with open('tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

Y_train = train['label']
Y_test = test['label']
Y_val = val['label']

Y_train = np.expand_dims(Y_train, [1])
Y_val = np.expand_dims(Y_val, [1])
Y_test = np.expand_dims(Y_test, [1])

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

loss_fn = tf.keras.losses.BinaryCrossentropy()
bce_metric = tf.keras.metrics.BinaryAccuracy()
optimizer = tf.keras.optimizers.Adam()
bce_val_metric = tf.keras.metrics.BinaryAccuracy()

pbar = tf.keras.utils.Progbar(len(train_dataset))

embedding_weights, match_counter = build_embeddings(
    len(tokenizer.word_index) + 1, tokenizer)

# model = build_multi_head_attention(
#     embedding_weights, len(tokenizer.word_index)+1, MAX_LEN, input_shape=(MAX_LEN,))

model = build_lstm_with_attention(
    embedding_weights, len(tokenizer.word_index)+1, MAX_LEN, input_shape=(MAX_LEN,))

weights_filenam = 'basic_attention_weights4'
#model = tf.keras.models.load_model(weights_filenam)
model.summary()


@tf.function
def train_step(X_train, Y_train, model):
    with tf.GradientTape() as tape:
        predictions = model(X_train, training=True)
        loss = loss_fn(Y_train, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    bce_metric.update_state(Y_train, predictions)
    return loss


@tf.function
def val_step(X_val, Y_val, model):
    predictions = model(X_val, training=False)
    loss = loss_fn(Y_val, predictions)
    bce_val_metric.update_state(Y_val, predictions)
    return loss


def train_loop():

    losses = []
    acc = []

    for e in range(EPOCHS):

        for j, (X_train, Y_train) in enumerate(train_dataset):
            loss = train_step(X_train, Y_train, model)
            pbar.update(j+1)

        losses.append(loss)
        acc.append(bce_metric.result())

        print(
            f"Training -- {e + 1} : loss:{loss}  accuracy:{bce_metric.result()}")
        bce_metric.reset_states()

        if e % 5 == 0:
            model.save(weights_filenam)

        for k, (X_val, Y_val) in enumerate(test_dataset):
            val_loss = val_step(X_val, Y_val, model)

        print(
            f"Validation --{e+1} loss:{val_loss} accuracy:{bce_val_metric.result()}")
        bce_val_metric.reset_states()

    model.save(weights_filenam)


def evaluate():
    test_bce_metric = tf.keras.metrics.BinaryAccuracy()
    acc = []
    for i, (X_test, Y_test) in enumerate(test_dataset):
        predictions = model.predict(X_test)
        test_bce_metric.update_state(Y_test, predictions)
        result = test_bce_metric.result()
        acc.append(result)
        print(f"Evalucation accurcy:{result}")
        test_bce_metric.reset_states()
    print(f"AVERAGE : {sum(acc)/ len(acc)}")


train_loop()
evaluate()
