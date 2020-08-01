import json
import os

from scipy import interpolate
import tensorflow as tf
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.python.keras.saving import load_model
import matplotlib.pyplot as plt

# Necessary for optimal performance on GPUs
from custom_code.data.dataset_builder import TFRecordsDatasetBuilder, Default2EnvBatchEqualizer, butter_filter

tf.compat.v1.enable_v2_behavior()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.keras.backend.set_session(session)


def initial_weight_difference():
    difference = {'spatial_difference': []}
    for i in range(1, 49):
        print("Model {}".format(i))
        if i < 10:
            subject_name = "subject_0{}".format(i)
        else:
            subject_name = "subject_{}".format(i)

        general_weights = load_model('output/general_models/' + subject_name + '_model.h5').get_weights()
        transfer_weights = load_model('output/transfer_models/' + subject_name + '_model.h5').get_weights()

        difference['spatial_difference'].append((transfer_weights[0] - general_weights[0]).tolist())
    with open("output/weight_difference.json", "w") as fp:
        json.dump(difference, fp)


def analyze():
    with open(os.path.join(os.getcwd(), "output", "weight_difference.json")) as fp:
        difference = json.loads(fp.read())
        spatial_difference = np.array(difference['spatial_difference']).reshape((48, 64, 6))
        plt.title("Heatmap of weight difference between general and transfer model")
        plt.imshow(sum(spatial_difference) / len(spatial_difference),
                   cmap='hot', interpolation='nearest')
        plt.ylabel("Frequency")
        plt.xlabel("Layer")
        plt.show()


def plot_weights(activations, save_dir):
    with open(os.path.join(os.getcwd(), "electrode_positions.json")) as fp:
        positions = json.loads(fp.read())
        x = positions['x_axis']
        y = positions['y_axis']

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8.5, 5))

        xi = np.linspace(-100, 100, 300)
        yi = np.linspace(-100, 100, 300)
        xy_center = [0, 0]
        for k, ax in zip(range(6), axes.flat):
            z = [i[k] for i in activations]
            zi = interpolate.griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
            dr = xi[1] - xi[0]
            for i in range(300):
                for j in range(300):
                    r = np.sqrt((xi[i] - xy_center[0]) ** 2 + (yi[j] - xy_center[1]) ** 2)
                    if (r - dr / 2) > 87.5352229800651:
                        zi[j, i] = "nan"
            sc = ax.contourf(xi, yi, zi, 60, cmap=plt.cm.jet, zorder=1, extend='both', vmin=-100, vmax=100,
                             levels=np.arange(-100, 100, 0.1))
            # ax.scatter(x, y, marker='o', c='b')
            ax.axis('off')

        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(sc, cax=cax, label="weight")
        fig.savefig(save_dir)


def __data_covariance(data_dir, save=False, load=False, window_size=128):
    tf.enable_eager_execution()
    ds_creator = TFRecordsDatasetBuilder(folder=data_dir)
    ds_train = ds_creator.prepare("train", batch_equalizer=Default2EnvBatchEqualizer(),
                                  batch_size=64, window=window_size)
    if load:
        with open(os.path.join(os.getcwd(), "output", "weight_difference.json")) as fp:
            data_covariance = json.loads(fp.read())
            return np.array(data_covariance[data_dir])

    data_covariance = {data_dir: None}
    total_entries = 0
    cov_sum = np.zeros((window_size, window_size))
    for ds in ds_train:
        ds_input, _ = ds
        eeg, _, _ = ds_input
        channels = eeg.numpy().T

        # Calculate covariance
        m1 = channels - channels.sum(2, keepdims=1) / window_size
        y_out = np.einsum('ijk,ilk->ijl', m1, m1) / (window_size - 1)

        cov_sum += np.sum(y_out, axis=0)
        total_entries += channels.shape[0]
    data_covariance[data_dir] = cov_sum / total_entries

    if save:
        with open("data_covariance.json", "w") as fp:
            data_covariance[data_dir] = data_covariance[data_dir].tolist()
            json.dump(data_covariance, fp)
    return data_covariance[data_dir]


def __latent_covariance(model_dir, data_dir, cutoff=None):
    ds_creator = TFRecordsDatasetBuilder(folder=data_dir)
    ds_train = ds_creator.prepare("train", batch_equalizer=Default2EnvBatchEqualizer(),
                                  batch_size=64, window=128)
    new_inputs = ()
    for ds in ds_train:
        ds_input, ds_output = ds
        eeg, good_env, bad_env = ds_input
        channels = eeg.numpy()[0]
        if cutoff is not None:
            cutoff_low, cutoff_high = cutoff
            channels = np.swapaxes(channels, 0, 1)
            channels = butter_filter(cutoff_low, cutoff_high, channels)
            channels = np.swapaxes(channels, 0, 1)
            new_inputs = (tf.convert_to_tensor([channels]), good_env, bad_env)
        break

    model = load_model(model_dir)
    intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                              outputs=model.layers[1].output)
    intermediate_output = intermediate_layer_model.predict(new_inputs)[0]
    return np.cov(intermediate_output.T)


def transfer_patterns(subject):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_folder = os.path.join(root, "dataset", "separated_data", subject)
    transfer_weights_i = np.array(load_model('output/transfer_models/' + subject + '_model.h5').get_weights()[0].tolist()[0])
    activations = np.dot(np.dot(__data_covariance(data_folder), transfer_weights_i),
                         __latent_covariance('output/transfer_models/' + subject + '_model.h5', data_folder))
    plot_weights(activations, 'plots/' + subject + '_transfer.png')


def general_patterns(subject):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_folder = os.path.join(root, "dataset", "separated_data", subject)
    transfer_weights_i = np.array(load_model('output/general_models/' + subject + '_model.h5').get_weights()[0].tolist()[0])
    activations = np.dot(np.dot(__data_covariance(data_folder), transfer_weights_i),
                         __latent_covariance('output/general_models/' + subject + '_model.h5', data_folder))
    plot_weights(activations, 'plots/' + subject + '_general.png')


def difference_patters(subject):
    with open(os.path.join(os.getcwd(), "output", "weight_difference.json")) as fp:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_folder = os.path.join(root, "dataset", "separated_data", subject)
        difference = json.loads(fp.read())
        spatial_difference = np.array(difference['spatial_difference']).reshape((48, 64, 6))
        activations = np.dot(np.dot(__data_covariance(data_folder), spatial_difference[0]),
                             __latent_covariance('output/transfer_models/' + subject + '_model.h5', data_folder))
        plot_weights(activations, 'plots/' + subject + '_difference.png')


if __name__ == "__main__":
    # initial_weight_difference()
    for i in range(1, 2):
        print("Model {}".format(i))
        if i < 10:
            subject_name = "subject_0{}".format(i)
        else:
            subject_name = "subject_{}".format(i)
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_folder = os.path.join(root, "dataset", "separated_data", subject_name)
        __data_covariance(data_folder, save=True)
        pass
        # general_patterns(subject_name)
        # transfer_patterns(subject_name)
        # difference_patters(subject_name)
