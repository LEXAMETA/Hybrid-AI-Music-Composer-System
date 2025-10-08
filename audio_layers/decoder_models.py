
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout, Reshape, Conv1D, LSTM, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class RecurrentAttention(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="glorot_uniform", trainable=True)
        self.V = self.add_weight(shape=(self.units, 1),
                                 initializer="glorot_uniform", trainable=True)
        super().build(input_shape)
    def call(self, x):
        h = tf.tanh(tf.tensordot(x, self.W, axes=1))
        score = tf.tensordot(h, self.V, axes=1)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * x, axis=1)
        return context_vector

def combined_time_freq_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    # You can add multi-resolution STFT loss or other time-frequency losses here
    return mse

def build_decoder_with_instincts(input_dim_raw, num_instincts=3, output_len=5512, lr=1e-4):
    """
    Builds the neural decoder model with instinct conditioning.
    Args:
        input_dim_raw (int): Dimension of raw input feature vector.
        num_instincts (int): Number of instincts (conditioning factors).
        output_len (int): Output length of synthesized audio waveform.
        lr (float): Learning rate for optimizer.
    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    inp_raw = Input(shape=(input_dim_raw,), name="raw_features")

    # Fusion weights predictor
    fusion_weights = Dense(num_instincts, activation="softmax", name="fusion_weights_predictor")(inp_raw)

    # Instinct conditioning layer
    # Expands inputs to multiple instincts with gating (simplified here):
    conditioned = tf.keras.layers.Lambda(
        lambda x: tf.stack([x] * num_instincts, axis=1),
        name="instinct_conditioning"
    )(inp_raw)

    # Weighting and fusion of instincted features:
    fused = tf.keras.layers.Dot(axes=1)([conditioned, tf.expand_dims(fusion_weights, -1)])

    # Core decoder network
    x = Dense(128, activation="relu")(fused)
    x = BatchNormalization()(x)
    x = Dropout(0.12)(x)

    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.12)(x)

    x = Dense(128 * 8, activation="relu")(x)
    x = Reshape((128, 8))(x)

    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    lstm_out = LSTM(128, return_sequences=True, dropout=0.05)(x)

    attention_out = RecurrentAttention(64)(lstm_out)

    x = Dense(512, activation="relu", kernel_regularizer=l2(1e-5))(attention_out)
    x = BatchNormalization()(x)

    out = Dense(output_len, activation="tanh")(x)

    model = Model(inputs=inp_raw, outputs=out, name="DecoderWithInstincts")

    adam = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    model.compile(optimizer=adam, loss=combined_time_freq_loss)

    return model
