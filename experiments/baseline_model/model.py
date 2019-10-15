import tensorflow as tf


def baseline_model(filters=1, kernel_size=16, time_window=640):
    eeg = tf.keras.layers.Input(shape=(time_window, 64))
    env1 = tf.keras.layers.Input(shape=(time_window, 1))
    env2 = tf.keras.layers.Input(shape=(time_window, 1))

    conv1d = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size)(eeg)
    env1_cut = tf.keras.layers.Lambda(lambda t: t[:, :time_window - kernel_size + 1, :])(env1)
    env2_cut = tf.keras.layers.Lambda(lambda t: t[:, :time_window - kernel_size + 1, :])(env2)

    dot1 = tf.keras.layers.Dot(0, normalize=True)([conv1d, env1_cut])
    dot2 = tf.keras.layers.Dot(0, normalize=True)([conv1d, env2_cut])

    concat = tf.keras.layers.Concatenate()([dot1, dot2])
    flat = tf.keras.layers.Flatten()(concat)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(flat)

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])
    model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["acc"], loss=["binary_crossentropy"])
    print(model.summary)
    return model
