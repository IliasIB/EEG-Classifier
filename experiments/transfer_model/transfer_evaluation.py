import os
import tensorflow as tf
import json

### Necessary for optimal performance on GPU's
tf.compat.v1.enable_v2_behavior()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.keras.backend.set_session(session)

from custom_code.data.dataset_builder import TFRecordsDatasetBuilder, Default2EnvBatchEqualizer


def evaluate():
    evaluation_transfer = {}
    evaluation_general = {}
    cwd = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for i in range(1, 54):
        if i == 8 or i == 19 or i == 22 or i == 26 or i == 27:
            continue
        if i < 10:
            subject_name = "subject_0{}".format(i)
        else:
            subject_name = "subject_{}".format(i)
        data_folder = os.path.join(root, "dataset", "separated_data", subject_name)

        ds_creator = TFRecordsDatasetBuilder(folder=data_folder)
        test_datasets = ds_creator.prepare("test", batch_equalizer=Default2EnvBatchEqualizer(),
                                           batch_size=64, window=128)

        # Evaluate transferred model
        general_model = tf.keras.models.load_model(os.path.join(cwd, "output", "general_models",
                                                                "{}_model.h5".format(subject_name)))
        # Evaluate general model
        for subject, ds_test in test_datasets.items():
            evaluation_general[subject] = dict(zip(general_model.metrics_names, general_model.evaluate(ds_test)))
            for k, v in evaluation_general[subject].items():
                evaluation_general[subject][k] = float(v)

        # Evaluate transferred model
        model = tf.keras.models.load_model(os.path.join(cwd, "output", "transfer_models",
                                                        "{}_model.h5".format(subject_name)))

        for subject, ds_test in test_datasets.items():
            evaluation_transfer[subject] = dict(zip(model.metrics_names, model.evaluate(ds_test)))
            for k, v in evaluation_transfer[subject].items():
                evaluation_transfer[subject][k] = float(v)

    with open("output/general_eval.json", "w") as fp:
        json.dump(evaluation_general, fp)
    with open("output/transfer_eval.json", "w") as fp:
        json.dump(evaluation_transfer, fp)


if __name__ == "__main__":
    evaluate()
