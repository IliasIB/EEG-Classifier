import json
import math
import os
import sys
import tensorflow as tf
from tensorflow.python.keras.saving import load_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import glorot_uniform

### Necessary for optimal performance on GPU's
tf.compat.v1.enable_v2_behavior()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.keras.backend.set_session(session)

from custom_code.keras.callbacks import StepCounter
from custom_code.data.dataset_builder import TFRecordsDatasetBuilder, Default2EnvBatchEqualizer
from experiments.sequence_lstm_model.model import simple_lstm_model


def filter(paths):
    # Just a sorting operation to make sure that every batch has data from different subjects
    return sorted(paths, key=lambda x: x.split("_-_")[-1])


def transfer_learn():
    for i in range(7, 8):
        if i == 8 or i == 19 or i == 22 or i == 26 or i == 27:
            continue
        if i < 10:
            subject_name = "subject_0{}".format(i)
        else:
            subject_name = "subject_{}".format(i)

        cwd = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_folder = os.path.join(root, "dataset", "separated_data", subject_name)
        ds_creator = TFRecordsDatasetBuilder(folder=data_folder)
        time_window = 128

        model = load_model('output/general_models/' + subject_name + '_model.h5')
        model.summary()

        model.get_layer(name="conv1d_17").trainable = False
        model.get_layer(name="bidirectional_5").trainable = False
        model.get_layer(name="cu_dnnlstm_11").trainable = False
        model.get_layer(name="dense_5").trainable = False

        model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["acc"], loss=["binary_crossentropy"])

        ds_train = ds_creator.prepare("train", batch_equalizer=Default2EnvBatchEqualizer(),
                                      batch_size=64, window=time_window)
        ds_validation = ds_creator.prepare("validation",
                                           batch_equalizer=Default2EnvBatchEqualizer(),
                                           batch_size=64, window=time_window)

        # TFRecords don't have metadata and depending on the building of the dataset the length might be different
        # To dynamically determine how many steps we should take, we let keras try to fit the model with a very high
        # number of steps. If data has run out, it will just issue a warning and continue. To count how many steps were
        # taken, the StepCounter is used
        train_s = StepCounter()
        model.fit(ds_train, epochs=1, steps_per_epoch=sys.maxsize, callbacks=[train_s])
        validation_s = StepCounter()
        model.fit(ds_validation, epochs=1, steps_per_epoch=sys.maxsize, callbacks=[validation_s])

        model.fit(
            ds_train.repeat(),
            epochs=50,
            steps_per_epoch=train_s.counter,
            validation_data=ds_validation.repeat(),
            validation_steps=validation_s.counter,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(os.path.join(cwd, "output", "transfer_models",
                                                                "{}_model.h5".format(subject_name)),
                                                   save_best_only=True),
                tf.keras.callbacks.CSVLogger(os.path.join(cwd, "output", "transfer_training.log")),
                tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", verbose=1, patience=2)
            ],
            initial_epoch=1
        )


def train_general_models():
    time_window = 128
    for i in range(2, 54):
        if i == 8 or i == 19 or i == 22 or i == 26 or i == 27:
            continue
        if i < 10:
            subject_name = "subject_0{}".format(i)
        else:
            subject_name = "subject_{}".format(i)

        ds_train, ds_validation = __build_unseen_dataset(i, time_window)

        # TFRecords don't have metadata and depending on the building of the dataset the length might be different
        # To dynamically determine how many steps we should take, we let keras try to fit the model with a very high
        # number of steps. If data has run out, it will just issue a warning and continue. To count how many steps were
        # taken, the StepCounter is used
        cwd = os.path.dirname(os.path.abspath(__file__))
        train_s = StepCounter()

        model = simple_lstm_model(time_window=time_window)
        model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["acc"], loss=["binary_crossentropy"])
        model.fit(ds_train, epochs=1, steps_per_epoch=sys.maxsize, callbacks=[train_s])
        validation_s = StepCounter()
        model.fit(ds_validation, epochs=1, steps_per_epoch=sys.maxsize, callbacks=[validation_s])

        model.fit(
            ds_train.repeat(),
            epochs=50,
            steps_per_epoch=train_s.counter,
            validation_data=ds_validation.repeat(),
            validation_steps=validation_s.counter,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(os.path.join(cwd, "output", "general_models",
                                                                "{}_model.h5".format(subject_name)),
                                                   save_best_only=True),
                tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", verbose=1, patience=2)
            ],
            initial_epoch=1
        )


def __build_unseen_dataset(i, time_window):
    if i == 1:
        tmp_name = "subject_0{}".format(2)
    else:
        tmp_name = "subject_0{}".format(1)
    ds_train, ds_validation = get_dataset(tmp_name, time_window)
    for j in range(1, 54):
        if j == 8 or j == 19 or j == 22 or j == 26 or j == 27 or i == j:
            continue
        if j < 10:
            subject_name_2 = "subject_0{}".format(j)
        else:
            subject_name_2 = "subject_{}".format(j)

        tmp_train, tmp_validation = get_dataset(subject_name_2, time_window)
        ds_train = ds_train.concatenate(tmp_train)
        ds_validation = ds_validation.concatenate(tmp_validation)
    return ds_train, ds_validation


def get_dataset(subject_name, time_window):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_folder = os.path.join(root, "dataset", "separated_data", subject_name)
    ds_creator = TFRecordsDatasetBuilder(folder=data_folder)
    ds_train = ds_creator.prepare("train", batch_equalizer=Default2EnvBatchEqualizer(),
                                  batch_size=64, window=time_window)
    ds_validation = ds_creator.prepare("validation",
                                       batch_equalizer=Default2EnvBatchEqualizer(),
                                       batch_size=64, window=time_window)
    return ds_train, ds_validation


if __name__ == "__main__":
    transfer_learn()
