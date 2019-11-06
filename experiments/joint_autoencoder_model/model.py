import tensorflow as tf


def joint_autoencoder_model(filters=5, kernel_size=16, time_window=640):
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

    model = tf.keras.Model(inputs=eeg, outputs=dec)
    model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["mse"], loss=["mse"])
    print(model.summary())
    return model
