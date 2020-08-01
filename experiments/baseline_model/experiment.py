import os
from custom_code.data.dataset_builder import Default2EnvBatchEqualizer
from experiments.baseline_model.model import baseline_model
from custom_code.tensorflow.helper import initialize, train_model

initialize()


def filter(paths):
    # Just a sorting operation to make sure that every batch has data from different subjects
    return sorted(paths, key=lambda x: x.split("_-_")[-1])


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    data_folder = os.path.join(root, "dataset", "full")

    time_window = 640
    batch_size = 64
    epochs = 20
    batch_equalizer = Default2EnvBatchEqualizer()
    model_location = os.path.join(cwd, "output", "model.h5")
    log_location = os.path.join(cwd, "output", "training.log")
    model = baseline_model(time_window=time_window)

    train_model(model, epochs, model_location, log_location, data_folder, batch_size, time_window, batch_equalizer)
