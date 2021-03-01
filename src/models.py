
import tensorflow as tf

EMBEDDING_DIMS = 200


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, num_heads, dims):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.dims = dims

        self.depth = self.dims // self.num_heads

        self.kw = tf.keras.layers.Dense(self.dims)
        self.qw = tf.keras.layers.Dense(self.dims)
        self.vw = tf.keras.layers.Dense(self.dims)

        self.dense = tf.keras.layers.Dense(self.dims)

    def split_heads(self, inputs):
        batch_size = tf.shape(inputs)[0]
        reshaped = tf.reshape(
            inputs, (batch_size, -1, self.num_heads, self.depth))
        transposed = tf.transpose(reshaped, perm=[0, 2, 1, 3])
        return transposed

    def call(self, k, v, q):
        k = self.kw(k)
        q = self.qw(q)
        v = self.vw(v)

        # multi head
        k = self.split_heads(k)
        q = self.split_heads(q)
        v = self.split_heads(v)

        # self attention

        mult_kq = tf.matmul(q, k, transpose_b=True)
        normalized_mult = mult_kq / tf.cast(self.depth, tf.float32)
        attention_weights = tf.nn.softmax(normalized_mult, axis=-1)
        output = tf.matmul(attention_weights, v)

        scaled_attention = tf.transpose(output, perm=[0, 2, 1, 3])

        batch_size = tf.shape(q)[0]
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.dims))
        output = self.dense(concat_attention)

        return output


class TokenAndPositionEmbeddings(tf.keras.layers.Layer):

    def __init__(self, maxlen, emb_dim, vocab_size, weights):
        super(TokenAndPositionEmbeddings, self).__init__()

        self.embds = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=emb_dim, weights=[weights])
        self.pos_embs = tf.keras.layers.Embedding(
            input_dim=maxlen, output_dim=emb_dim)

    def call(self, inputs):
        max_len = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_embs(positions)
        embeddings = self.embds(inputs)
        return embeddings + positions


class Transformer(tf.keras.layers.Layer):
    """
    Transfomer Encoder with Multi-head attention (6 heads)
    """

    def __init__(self, dims, num_heads, ff_dim, rate=0.1):
        super(Transformer, self).__init__()

        self.mha = MultiHeadAttention(num_heads, dims)
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(dims)
        ])

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attention_output = self.mha(inputs, inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        layer_norm1 = self.layer_norm1(inputs + attention_output)

        ff_output = self.ff(layer_norm1)
        ff_output = self.dropout2(ff_output, training=training)
        return self.layer_norm2(layer_norm1 + ff_output)


def build_multi_head_attention(embedding_weights, vocab_size, max_len, input_shape=(None,)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = TokenAndPositionEmbeddings(
        max_len, EMBEDDING_DIMS, vocab_size, embedding_weights)(inputs)
    x = Transformer(EMBEDDING_DIMS, num_heads=4, ff_dim=512)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1, name="drop1")(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1, name="drop2")(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


# basic attention

class BasicAttention(tf.keras.layers.Layer):

    def __init__(self, units):
        super(BasicAttention, self).__init__()

        self.w1 = tf.keras.layers.Dense(units)
        self.w2 = tf.keras.layers.Dense(units)

        self.v = tf.keras.layers.Dense(1)

    def call(self, features, hidden_state):

        hidden_state_with_axis = tf.expand_dims(hidden_state, 1)
        scores = tf.nn.tanh(self.w1(features) +
                            self.w2(hidden_state_with_axis))
        attention_weights = tf.nn.softmax(self.v(scores), axis=1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


def build_lstm_with_attention(embedding_weights, vocab_size, max_len, input_shape=(None,)):
    """
    Bi-directional LSTM with Attention
    """
    inp = tf.keras.layers.Input(shape=input_shape)
    embedding = tf.keras.layers.Embedding(
        vocab_size,
        output_dim=EMBEDDING_DIMS,
        weights=[embedding_weights],
        input_length=max_len
    )(inp)

    lstm1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, activation='tanh')
    )(embedding)

    (lstm, forward_h, forward_c, backward_h, backward_d) = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)
    )(lstm1)

    hidden_state = tf.keras.layers.Concatenate()([forward_h, backward_h])
    context_vector, attention_weights = BasicAttention(16)(lstm, hidden_state)

    dense1 = tf.keras.layers.Dense(64, activation='relu')(context_vector)
    dropout = tf.keras.layers.Dropout(0.5)(dense1)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)

    model = tf.keras.Model(inputs=inp, outputs=output)
    return model
