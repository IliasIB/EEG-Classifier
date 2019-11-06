import os
import tensorflow as tf
import json


from custom_code.data.dataset_builder import TFRecordsDatasetBuilder, Encoder2EnvBatchEqualizer


def evaluate():
    cwd = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_folder = os.path.join(root, "dataset", "starting_data")


    evaluation = {}

    ds_creator = TFRecordsDatasetBuilder(folder=data_folder)
    test_datasets = ds_creator.prepare("test", batch_equalizer=Encoder2EnvBatchEqualizer(),
                                       batch_size=64, window=100)

    model = tf.keras.models.load_model(os.path.join(cwd, "output", "best_model.h5"))

    for subject, ds_test in test_datasets.items():
        evaluation[subject] = dict(zip(model.metrics_names, model.evaluate(ds_test)))
        for k, v in evaluation[subject].items():
            evaluation[subject][k] = float(v)

    with open("output/eval.json", "w") as fp:
        json.dump(evaluation, fp)


if __name__ == "__main__":
    evaluate()
