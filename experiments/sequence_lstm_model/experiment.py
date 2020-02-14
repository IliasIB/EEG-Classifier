import json
import math
import os
import sys
import tensorflow as tf

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


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_folder = os.path.join(root, "dataset", "starting_data")
    ds_creator = TFRecordsDatasetBuilder(folder=data_folder)
    time_window = 128

    model = simple_lstm_model(time_window=time_window)

    ds_train = ds_creator.prepare("train", batch_equalizer=Default2EnvBatchEqualizer(),
                                  batch_size=64, window=time_window)
    ds_validation = ds_creator.prepare("validation",
                                       batch_equalizer=Default2EnvBatchEqualizer(),
                                       batch_size=64, window=time_window)

    # TFRecords don't have metadata and depending on the building of the dataset the length might be different
    # To dynamically determine how many steps we should take, we let keras try to fit the model with a very high number
    # of steps. If data has run out, it will just issue a warning and continue. To count how many steps were taken, the
    # StepCounter is used
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
            tf.keras.callbacks.ModelCheckpoint(os.path.join(cwd, "output", "two_second_model.h5"), save_best_only=True),
            tf.keras.callbacks.CSVLogger(os.path.join(cwd, "output", "training.log")),
            tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", verbose=1, patience=4)
        ],
        initial_epoch=1
    )
