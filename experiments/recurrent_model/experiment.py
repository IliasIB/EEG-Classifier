import argparse
import os
from custom_code.data.dataset_builder import Default2EnvBatchEqualizer
from experiments.recurrent_model.model import recurrent_model
from custom_code.tensorflow.helper import initialize, train_model

initialize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the recurrent model')
    parser.add_argument('--window', '--w', metavar='TIME_WINDOW', type=int,
                        help='size of window to train the model on', required=True)
    parser.add_argument('--epochs', '--e', metavar='EPOCHS', type=int,
                        help='amount of epochs to train the model', required=True)
    parser.add_argument('--batch', '--b', metavar='BATCH_SIZE', type=int,
                        help='size of batch to use during training', required=True)
    parser.add_argument('--band', '--B', metavar='BAND', type=str, choices=['full', 'delta', 'theta', 'delta_theta'],
                        help='which band(s) to use for training', required=True)
    args = parser.parse_args()

    cwd = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    train_model(model=recurrent_model(time_window=args.window), epochs=args.epochs,
                model_location=os.path.join(cwd, "output", "model_{}_{}.h5".format(args.window, args.band)),
                log_location=os.path.join(cwd, "output", "training_{}_{}.log".format(args.window, args.band)),
                data_folder=os.path.join(root, "dataset", args.band), batch_size=args.batch, time_window=args.window,
                batch_equalizer=Default2EnvBatchEqualizer())
