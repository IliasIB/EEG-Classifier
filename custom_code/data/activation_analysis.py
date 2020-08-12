import json
import os
from scipy import interpolate
import tensorflow as tf
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from custom_code.data.dataset_builder import TFRecordsDatasetBuilder, Default2EnvBatchEqualizer

tf.compat.v1.enable_v2_behavior()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.keras.backend.set_session(session)


def plot_activations(activations, save_location):
    with open(os.path.join(os.getcwd(), "electrode_positions.json")) as fp:
        positions = json.loads(fp.read())
        x = positions['x_axis']
        y = positions['y_axis']

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8.5, 5))

        xi, yi = np.linspace(-100, 100, 300), np.linspace(-100, 100, 300)
        for k, ax in zip(range(6), axes.flat):
            z = [i[k] for i in activations]
            zi = interpolate.griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
            for i in range(300):
                for j in range(300):
                    if (np.sqrt(xi[i] ** 2 + yi[j] ** 2) - (xi[1] - xi[0]) / 2) > 87.5352229800651:
                        zi[j, i] = "nan"
            sc = ax.contourf(xi, yi, zi, 60, cmap=plt.cm.jet, zorder=1, extend='both', vmin=-100, vmax=100,
                             levels=np.arange(-100, 100, 0.1))
            ax.axis('off')

        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(sc, cax=cax, label="activation")
        plt.show()
        fig.savefig(save_location)


def activation_pattern(data_dir, used_model, mode, window_size=128, batch_size=64, save=False, load=False):
    if load:
        with open(os.path.join(os.getcwd(), "activation.json")) as fp:
            activations = json.loads(fp.read())
        if not save:
            return np.array(activations[mode])
    else:
        try:
            with open(os.path.join(os.getcwd(), "activation.json")) as fp:
                activations = json.loads(fp.read())
        except FileNotFoundError:
            activations = {mode: None}
    tf.enable_eager_execution()
    ds_creator = TFRecordsDatasetBuilder(folder=data_dir)
    dataset = ds_creator.prepare("train", batch_size=batch_size, window=window_size)
    intermediate_layer_model = tf.keras.Model(inputs=used_model.input,
                                              outputs=used_model.layers[1].output)
    weights = np.array(used_model.get_weights()[0].tolist()[0])

    total_entries = 0
    activation_sum = np.zeros((weights.shape[0], weights.shape[1]))
    for ds in dataset:
        eeg, _, _ = ds
        channels = eeg.numpy()
        channels = np.swapaxes(channels, 2, 1)
        repeated_weights = np.repeat([weights], channels.shape[0], axis=0)

        intermediate_output = intermediate_layer_model.predict(ds)
        intermediate_output = np.swapaxes(intermediate_output, 2, 1)

        activation_batch = __activation_batch(channels, intermediate_output, repeated_weights, window_size)

        activation_sum += np.sum(activation_batch, axis=0)
        total_entries += channels.shape[0]
    activations[mode] = activation_sum / total_entries

    if save:
        with open("activation.json", "w") as fp:
            activations[mode] = activations[mode].tolist()
            json.dump(activations, fp)
    return activations[mode]


def __activation_batch(channels, intermediate_output, weights, window_size):
    # Calculate data covariance
    m1 = channels - channels.sum(2, keepdims=1) / window_size
    data_cov = np.einsum('ijk,ilk->ijl', m1, m1) / (window_size - 1)

    # Calculate latent covariance
    m1 = intermediate_output - intermediate_output.sum(2, keepdims=1) / window_size
    latent_cov = np.einsum('ijk,ilk->ijl', m1, m1) / (window_size - 1)

    # Calculate activation
    activation_batch = np.matmul(np.matmul(data_cov, weights), latent_cov)

    return activation_batch
