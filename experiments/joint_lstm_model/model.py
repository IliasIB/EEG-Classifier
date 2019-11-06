import os

import tensorflow as tf


def joint_lstm_model(filters=1, kernel_size=16, time_window=640):
    # Autoencoder model
    cwd = os.path.dirname(os.path.abspath(__file__))
    model = tf.keras.models.load_model(os.path.join(cwd, "..", "joint_autoencoder_model",
                                                    "output", "best_model.h5"))

    # Input
    eeg = model.input
    env1 = tf.keras.layers.Input(shape=(time_window, 1), name="input_env1")
    env2 = tf.keras.layers.Input(shape=(time_window, 1), name="input_env2")

    encoder = model.get_layer("max_pooling1d_1").output
    enc_dense = tf.keras.layers.Dense(1, activation="sigmoid")(encoder)

    flat1 = tf.keras.layers.Flatten()(env1)
    flat2 = tf.keras.layers.Flatten()(env2)
    env1_dense = tf.keras.layers.Dense(160, activation="sigmoid")(flat1)
    env2_dense = tf.keras.layers.Dense(160, activation="sigmoid")(flat2)

    dot1 = tf.keras.layers.Dot(0, normalize=True)([enc_dense, env1_dense])
    dot2 = tf.keras.layers.Dot(0, normalize=True)([enc_dense, env2_dense])

    concat = tf.keras.layers.Concatenate()([dot1, dot2])
    flat = tf.keras.layers.Flatten()(concat)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(flat)

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])
    model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["acc"], loss=["binary_crossentropy"])
    print(model.summary())
    return model
