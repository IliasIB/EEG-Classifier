import os
import sys
import tensorflow as tf
from tensorflow.python.keras.saving import load_model
from custom_code.keras.callbacks import StepCounter
from custom_code.data.dataset_builder import TFRecordsDatasetBuilder, Default2EnvBatchEqualizer, subjects
from custom_code.tensorflow.helper import initialize, train_model
from experiments.recurrent_model.model import recurrent_model

initialize()


def filter(paths):
    # Just a sorting operation to make sure that every batch has data from different subjects
    return sorted(paths, key=lambda x: x.split("_-_")[-1])


def transfer_learn():
    for i in range(1, 49):
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

        model.layers[5].trainable = False
        model.layers[6].trainable = False
        model.layers[7].trainable = False
        model.layers[12].trainable = False

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
            epochs=100,
            steps_per_epoch=train_s.counter,
            validation_data=ds_validation.repeat(),
            validation_steps=validation_s.counter,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(os.path.join(cwd, "output", "transfer_models",
                                                                "{}_model.h5".format(subject_name)),
                                                   save_best_only=True),
                tf.keras.callbacks.CSVLogger(os.path.join(cwd, "output", "training_logs",
                                                          "transfer_training{}.log".format(str(i)))),
                tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", verbose=1, patience=15)
            ],
            initial_epoch=1
        )


def train_general_models():
    time_window = 128
    for i in range(1, 49):
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

        model = recurrent_model(time_window=time_window)
        model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["acc"], loss=["binary_crossentropy"])
        model.fit(ds_train, epochs=1, steps_per_epoch=sys.maxsize, callbacks=[train_s])
        validation_s = StepCounter()
        model.fit(ds_validation, epochs=1, steps_per_epoch=sys.maxsize, callbacks=[validation_s])

        model.fit(
            ds_train.repeat(),
            epochs=2,
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
    for j in range(1, 49):
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
    for name in subjects:
        cwd = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_folder = os.path.join(root, "dataset", "full")

        def subject_filter(paths):
            # Just a sorting operation to make sure that every batch has data from different subjects
            return [path for path in paths if path.split("_-_")[1] != name]

        time_window = 640
        batch_size = 64
        epochs = 20
        batch_equalizer = Default2EnvBatchEqualizer()
        model_location = os.path.join(cwd, "output", "model_general_{}.h5".format(name))
        log_location = os.path.join(cwd, "output", "training_general_{}.log".format(name))
        model = recurrent_model(time_window=time_window)

        train_model(model, epochs, model_location, log_location, data_folder, batch_size, time_window, batch_equalizer,
                    [subject_filter])

        # Transfer learning
        model.layers[5].trainable = False
        model.layers[6].trainable = False
        model.layers[7].trainable = False
        model.layers[12].trainable = False

        def transfer_filter(paths):
            # Just a sorting operation to make sure that every batch has data from different subjects
            return [path for path in paths if path.split("_-_")[1] == name]


        transfer_model_location = os.path.join(cwd, "output", "model_transfer_{}.h5".format(name))
        transfer_log_location = os.path.join(cwd, "output", "training_transfer_{}.log".format(name))

        train_model(model, epochs, model_location, transfer_model_location, transfer_log_location, batch_size,
                    time_window, batch_equalizer, [transfer_filter])
