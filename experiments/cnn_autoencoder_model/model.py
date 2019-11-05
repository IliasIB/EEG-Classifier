import tensorflow as tf


def cnn_autoencoder_model(filters=5, kernel_size=16, time_window=640):
    eeg = tf.keras.layers.Input(shape=(time_window, 64))
    env1 = tf.keras.layers.Input(shape=(time_window, 1))
    env2 = tf.keras.layers.Input(shape=(time_window, 1))

    # Encoder
    enc = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same")(eeg)
    enc1 = tf.keras.layers.MaxPooling1D(2, padding="same")(enc)
    enc2 = tf.keras.layers.Conv1D(16, kernel_size=3, padding="same")(enc1)
    enc3 = tf.keras.layers.MaxPooling1D(2, padding="same")(enc2)

    # Decoder
    dec4 = tf.keras.layers.Conv1D(16, kernel_size=3, padding="same")(enc3)
    dec3 = tf.keras.layers.UpSampling1D(2)(dec4)
    dec2 = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same")(dec3)
    dec1 = tf.keras.layers.UpSampling1D(2)(dec2)
    dec = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same")(dec1)

    dot1 = tf.keras.layers.Dot(0, normalize=True)([dec, env1])
    dot2 = tf.keras.layers.Dot(0, normalize=True)([dec, env2])

    concat = tf.keras.layers.Concatenate()([dot1, dot2])
    flat = tf.keras.layers.Flatten()(concat)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(flat)

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])
    model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["acc"], loss=["binary_crossentropy"])
    print(model.summary())
    return model
