import os

import tensorflow as tf


def full_joint_model(filters=1, kernel_size=16, time_window=640):
    # Autoencoder model
    eeg = tf.keras.layers.Input(shape=(time_window, 64))
    env1 = tf.keras.layers.Input(shape=(time_window, 1))
    env2 = tf.keras.layers.Input(shape=(time_window, 1))

    # Encoder
    enc = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same")(eeg)
    enc1 = tf.keras.layers.MaxPooling1D(2, padding="same")(enc)
    enc2 = tf.keras.layers.Conv1D(16, kernel_size=3, padding="same")(enc1)
    enc3 = tf.keras.layers.MaxPooling1D(2, padding="same")(enc2)
    enc_drop = tf.keras.layers.Dropout(0.2)(enc3)
    # # Decoder
    # dec4 = tf.keras.layers.Conv1D(16, kernel_size=3, padding="same")(enc3)
    # dec3 = tf.keras.layers.UpSampling1D(2)(dec4)
    # dec2 = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same")(dec3)
    # dec1 = tf.keras.layers.UpSampling1D(2)(dec2)
    # dec = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same")(dec1)

    # Input
    enc_dense = tf.keras.layers.Dense(1, activation="sigmoid")(enc_drop)

    flat1 = tf.keras.layers.Flatten()(env1)
    flat2 = tf.keras.layers.Flatten()(env2)
    env1_dense = tf.keras.layers.Dense(160, activation="sigmoid")(flat1)
    env2_dense = tf.keras.layers.Dense(160, activation="sigmoid")(flat2)

    dot1 = tf.keras.layers.Dot(0, normalize=True)([enc_dense, env1_dense])
    dot2 = tf.keras.layers.Dot(0, normalize=True)([enc_dense, env2_dense])

    concat = tf.keras.layers.Concatenate()([dot1, dot2])
    flat = tf.keras.layers.Flatten()(concat)
    flat_drop = tf.keras.layers.Dropout(0.2)(flat)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(flat_drop)

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])
    model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["acc"], loss=["binary_crossentropy"])
    print(model.summary())
    return model
