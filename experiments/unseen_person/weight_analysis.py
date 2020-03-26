import json
import math
import os
import sys
import tensorflow as tf
import numpy as np
import pandas
from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import glorot_uniform
import matplotlib.pyplot as plt

### Necessary for optimal performance on GPU's
tf.compat.v1.enable_v2_behavior()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.keras.backend.set_session(session)

from custom_code.keras.callbacks import StepCounter
from custom_code.data.dataset_builder import TFRecordsDatasetBuilder, Default2EnvBatchEqualizer
from experiments.sequence_lstm_model.model import simple_lstm_model


def initial_weight_difference():
    difference = {'spatial_difference': [], 'temporal_difference': []}
    for i in range(1, 54):
        print("Model {}".format(i))
        if i == 8 or i == 19 or i == 22 or i == 26 or i == 27:
            continue
        if i < 10:
            subject_name = "subject_0{}".format(i)
        else:
            subject_name = "subject_{}".format(i)

        general_weights = load_model('output/general_models/' + subject_name + '_model.h5').get_weights()
        transfer_weights = load_model('output/transfer_models/' + subject_name + '_model.h5').get_weights()

        difference['spatial_difference'].append((transfer_weights[0] - general_weights[0]).tolist())
        difference['temporal_difference'].append((transfer_weights[2] - general_weights[2]).tolist())
    with open("output/weight_difference.json", "w") as fp:
        json.dump(difference, fp)


def analyze():
    with open(os.path.join(os.getcwd(), "output", "weight_difference.json")) as fp:
        difference = json.loads(fp.read())
        spatial_difference = np.array(difference['spatial_difference']).reshape((48, 64, 6))
        # temporal_difference = np.array(difference['temporal_difference']).reshape((48, 64, 6))
        plt.title("Heatmap of weight difference between general and transfer model")
        plt.imshow(sum(spatial_difference) / len(spatial_difference),
                   cmap='hot', interpolation='nearest')
        plt.ylabel("Frequency")
        plt.xlabel("Layer")
        plt.show()
        # plt.imshow(sum(temporal_difference) / len(temporal_difference),
        #            cmap='hot', interpolation='nearest')


if __name__ == "__main__":
    analyze()
