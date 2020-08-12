import argparse
import os
import tensorflow as tf
from custom_code.tensorflow.helper import initialize, test_model
from custom_code.data.dataset_builder import Default2EnvBatchEqualizer

initialize()


def evaluate():
    parser = argparse.ArgumentParser(description='Evaluate the recurrent model')
    parser.add_argument('--window', '--w', metavar='TIME_WINDOW', type=int,
                        help='size of window to train the model on', required=True)
    parser.add_argument('--batch', '--b', metavar='BATCH_SIZE', type=int,
                        help='size of batch to use during training', required=True)
    parser.add_argument('--band', '--B', metavar='BAND', type=str, choices=['full', 'delta', 'theta', 'delta_theta'],
                        help='which band(s) to use for testing', required=True)
    args = parser.parse_args()

    cwd = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    test_model(model=tf.keras.models.load_model(os.path.join(cwd, "output",
                                                             "model_{}_{}.h5".format(args.window, args.band))),
               eval_location="output/evaluation_{}_{}.json".format(args.window, args.band),
               data_folder=os.path.join(root, "dataset", args.band), batch_size=args.batch, time_window=args.window,
               batch_equalizer=Default2EnvBatchEqualizer())


if __name__ == "__main__":
    evaluate()
