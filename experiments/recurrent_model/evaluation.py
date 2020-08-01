import os
import tensorflow as tf
from custom_code.tensorflow.helper import initialize, test_model
from custom_code.data.dataset_builder import Default2EnvBatchEqualizer

initialize()


def evaluate():
    cwd = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_folder = os.path.join(root, "dataset", "separated")

    eval_location = "output/eval.json"
    batch_size = 128
    time_window = 640
    batch_equalizer = Default2EnvBatchEqualizer()
    model = tf.keras.models.load_model(os.path.join(cwd, "output", "best_model.h5"))

    test_model(model, eval_location, data_folder, batch_size, time_window, batch_equalizer)


if __name__ == "__main__":
    evaluate()
