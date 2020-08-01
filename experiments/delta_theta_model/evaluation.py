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
    for mode in ['starting_data', 'tf_48_delta', 'tf_48_theta', 'tf_48_delta_theta']:
        cwd = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_folder = os.path.join(root, "dataset", mode)

        evaluation = {}

        ds_creator = TFRecordsDatasetBuilder(folder=data_folder)
        test_datasets = ds_creator.prepare("test", batch_equalizer=Default2EnvBatchEqualizer(),
                                           batch_size=64, window=128)

        model = tf.keras.models.load_model(os.path.join(cwd, "models", "{}.h5".format(mode)))

        for subject, ds_test in test_datasets.items():
            evaluation[subject] = dict(zip(model.metrics_names, model.evaluate(ds_test)))
            for k, v in evaluation[subject].items():
                evaluation[subject][k] = float(v)

        with open("evaluations/{}.json".format(mode), "w") as fp:
            json.dump(evaluation, fp)


if __name__ == "__main__":
    evaluate()
