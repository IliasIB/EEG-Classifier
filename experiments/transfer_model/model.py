import tensorflow as tf


def simple_lstm_model(filters=1, kernel_size=16, time_window=640):
    eeg = tf.keras.layers.Input(shape=(time_window, 64))
    env1 = tf.keras.layers.Input(shape=(time_window, 1))
    env2 = tf.keras.layers.Input(shape=(time_window, 1))

    conv1d = tf.keras.layers.Conv1D(6, strides=1, kernel_size=1)(eeg)
    conv1d2 = tf.keras.layers.Conv1D(24, strides=2, kernel_size=kernel_size, activation="relu")(conv1d)
    conv1d_layer = tf.keras.layers.Conv1D(24, strides=2, kernel_size=kernel_size, activation="relu")
    conv1d_env1 = conv1d_layer(env1)
    conv1d_env2 = conv1d_layer(env2)

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(16, return_sequences=True))(conv1d2)
    lstm_layer = tf.keras.layers.CuDNNLSTM(32, return_sequences=True)
    lstm_env1 = lstm_layer(conv1d_env1)
    lstm_env2 = lstm_layer(conv1d_env2)

    dot1 = tf.keras.layers.Dot(1, normalize=True)([lstm, lstm_env1])
    dot2 = tf.keras.layers.Dot(1, normalize=True)([lstm, lstm_env2])

    concat = tf.keras.layers.Concatenate()([dot1, dot2])
    flat = tf.keras.layers.Flatten()(concat)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(flat)

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])
    model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["acc"], loss=["binary_crossentropy"])
    print(model.summary())
    return model
