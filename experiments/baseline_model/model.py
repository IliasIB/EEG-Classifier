import tensorflow as tf


def baseline_model(filters=1, kernel_size=16, time_window=640):
    eeg = tf.keras.layers.Input(shape=(time_window, 64), name="eeg_input")
    env1 = tf.keras.layers.Input(shape=(time_window, 1), name="env1_input")
    env2 = tf.keras.layers.Input(shape=(time_window, 1), name="env2_input")

    conv1d = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, name="conv1d")(eeg)
    env1_cut = tf.keras.layers.Lambda(lambda t: t[:, :time_window - kernel_size + 1, :], name="cut_env1")(env1)
    env2_cut = tf.keras.layers.Lambda(lambda t: t[:, :time_window - kernel_size + 1, :], name="cut_env2")(env2)

    dot1 = tf.keras.layers.Dot(0, normalize=True, name="cos_sim_env1")([conv1d, env1_cut])
    dot2 = tf.keras.layers.Dot(0, normalize=True, name="cos_sim_env2")([conv1d, env2_cut])

    concat = tf.keras.layers.Concatenate(name="concat")([dot1, dot2])
    flat = tf.keras.layers.Flatten(name="flat")(concat)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="sigmoid")(flat)

    model = tf.keras.Model(inputs=[eeg, env1, env2], outputs=[out])
    model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["acc"], loss=["binary_crossentropy"])
    print(model.summary())
    return model
