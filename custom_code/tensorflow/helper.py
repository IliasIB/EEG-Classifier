import sys
import json
import tensorflow as tf
from custom_code.keras.callbacks import StepCounter
from custom_code.data.dataset_builder import TFRecordsDatasetBuilder


def initialize():
    tf.compat.v1.enable_v2_behavior()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.keras.backend.set_session(session)


def ds_builder(data_folder, batch_size, time_window, batch_equalizer, filters=tuple()):
    ds_creator = TFRecordsDatasetBuilder(folder=data_folder, filters=filters)
    ds_train = ds_creator.prepare("train", batch_equalizer=batch_equalizer,
                                  batch_size=batch_size, window=time_window)
    ds_validation = ds_creator.prepare("validation",
                                       batch_equalizer=batch_equalizer,
                                       batch_size=batch_size, window=time_window)
    return ds_train, ds_validation


def train_model(model, epochs, model_location, log_location, data_folder, batch_size, time_window,
                batch_equalizer, filters=tuple()):
    ds_train, ds_validation = ds_builder(data_folder, batch_size, time_window, batch_equalizer, filters)

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
        epochs=epochs,
        steps_per_epoch=train_s.counter,
        validation_data=ds_validation.repeat(),
        validation_steps=validation_s.counter,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(model_location, save_best_only=True),
            tf.keras.callbacks.CSVLogger(log_location),
            tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", verbose=1, patience=5)
        ],
        initial_epoch=1
    )


def test_model(model, eval_location, data_folder, batch_size, time_window, batch_equalizer):
    evaluation = {}

    ds_creator = TFRecordsDatasetBuilder(folder=data_folder)
    test_dataset = ds_creator.prepare("test", batch_equalizer=batch_equalizer,
                                      batch_size=batch_size, window=time_window)

    for subject, ds_test in test_dataset.items():
        evaluation[subject] = dict(zip(model.metrics_names, model.evaluate(ds_test)))
        for k, v in evaluation[subject].items():
            evaluation[subject][k] = float(v)

    with open(eval_location, "w") as fp:
        json.dump(evaluation, fp)
