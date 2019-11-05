import tensorflow as tf


def autoencoder_lstm_model(filters=1, kernel_size=16, time_window=640):
    eeg = tf.keras.layers.Input(shape=(time_window, 64))
    env1 = tf.keras.layers.Input(shape=(time_window, 1))
    env2 = tf.keras.layers.Input(shape=(time_window, 1))

    lstm_encoder = tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(time_window, 64))(eeg)
    lstm_encoder2 = tf.keras.layers.LSTM(16, input_shape=(time_window, 16))(lstm_encoder)
    lstm_decoder_repeat = tf.keras.layers.RepeatVector(time_window)(lstm_encoder2)
    lstm_decoder1 = tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(time_window, 16))(lstm_decoder_repeat)
    lstm_decoder2 = tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(time_window, 64))(lstm_decoder1)
    lstm_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(lstm_decoder2)

    dot1 = tf.keras.layers.Dot(0, normalize=True)([lstm_out, env1])
    dot2 = tf.keras.layers.Dot(0, normalize=True)([lstm_out, env2])

    concat = tf.keras.layers.Concatenate()([dot1, dot2])
    flat = tf.keras.layers.Flatten()(concat)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(flat)

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])
    model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["acc"], loss=["binary_crossentropy"])
    print(model.summary())
    return model
