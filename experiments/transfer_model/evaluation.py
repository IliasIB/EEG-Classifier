import argparse
import os
import json
import tensorflow as tf
from custom_code.tensorflow.helper import initialize, test_model
from custom_code.data.dataset_builder import Default2EnvBatchEqualizer, subjects

initialize()


def evaluate_general():
    evaluation = {}
    for name in subjects:
        def subject_filter(paths):
            return [path for path in paths if path.split("_-_")[1] != name]

        general_model = tf.keras.models.load_model(os.path.join(cwd, "output"
                                                                     "model_general_{}_{}_{}.h5".format(name,
                                                                                                        args.window,
                                                                                                        args.band)))
        single_eval = test_model(model=general_model, eval_location="", data_folder=data_folder, batch_size=args.batch,
                                 time_window=args.window, batch_equalizer=Default2EnvBatchEqualizer(),
                                 filters=[subject_filter], save=False)

        evaluation[name] = single_eval[name]
    with open("output/evaluation_general_{}_{}.json".format(args.window, args.band), "w") as fp:
        json.dump(evaluation, fp)


def evaluate_transfer():
    evaluation = {}
    for name in subjects:
        def transfer_filter(paths):
            return [path for path in paths if path.split("_-_")[1] == name]

        transfer_model = tf.keras.models.load_model(os.path.join(cwd, "output"
                                                                      "model_transfer_{}_{}_{}.h5".format(name,
                                                                                                          args.window,
                                                                                                          args.band)))
        single_eval = test_model(model=transfer_model, eval_location="", data_folder=data_folder, batch_size=args.batch,
                                 time_window=args.window, batch_equalizer=Default2EnvBatchEqualizer(),
                                 filters=[transfer_filter], save=False)

        evaluation[name] = single_eval[name]
    with open("output/evaluation_transfer_{}_{}.json".format(args.window, args.band), "w") as fp:
        json.dump(evaluation, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the subject-specific general and transfer models')
    parser.add_argument('--window', '--w', metavar='TIME_WINDOW', type=int,
                        help='size of window to train the model on', required=True)
    parser.add_argument('--batch', '--b', metavar='BATCH_SIZE', type=int,
                        help='size of batch to use during training', required=True)
    parser.add_argument('--band', '--B', metavar='BAND', type=str, choices=['full', 'delta', 'theta', 'delta_theta'],
                        help='which band(s) to use for training', required=True)
    args = parser.parse_args()

    cwd = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_folder = os.path.join(root, "dataset", args.band)

    evaluate_general()
    evaluate_transfer()
