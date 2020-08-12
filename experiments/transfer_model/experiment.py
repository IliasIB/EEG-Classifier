import argparse
import os
from custom_code.data.dataset_builder import Default2EnvBatchEqualizer, subjects
from custom_code.tensorflow.helper import initialize, train_model
from experiments.recurrent_model.model import recurrent_model

initialize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the recurrent model in a one-subject left--out fashion and '
                                                 'transfer the left-out subject')
    parser.add_argument('--window', '--w', metavar='TIME_WINDOW', type=int,
                        help='size of window to train the model on', required=True)
    parser.add_argument('--epochs', '--e', metavar='EPOCHS', type=int,
                        help='amount of epochs to train the model', required=True)
    parser.add_argument('--batch', '--b', metavar='BATCH_SIZE', type=int,
                        help='size of batch to use during training', required=True)
    parser.add_argument('--band', '--B', metavar='BAND', type=str, choices=['full', 'delta', 'theta', 'delta_theta'],
                        help='which band(s) to use for training', required=True)
    args = parser.parse_args()

    for name in subjects:
        cwd = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_folder = os.path.join(root, "dataset", args.band)

        def subject_filter(paths):
            return [path for path in paths if path.split("_-_")[1] != name]

        model = recurrent_model(time_window=args.window)

        train_model(model=model, epochs=args.epochs,
                    model_location=os.path.join(cwd, "output", "model_general_{}_{}_{}.h5".format(name, args.window,
                                                                                                  args.band)),
                    log_location=os.path.join(cwd, "output", "training_general_{}_{}_{}.log".format(name, args.window,
                                                                                                    args.band)),
                    data_folder=data_folder, batch_size=args.batch, time_window=args.window,
                    batch_equalizer=Default2EnvBatchEqualizer(), filters=[subject_filter])

        # Transfer learning
        model.get_layer(name="eeg_conv1d_space").trainable = False
        model.get_layer(name="eeg_bilstm").trainable = False
        model.get_layer(name="env_conv1d").trainable = False
        model.get_layer(name="env_lstm").trainable = False
        model.get_layer(name="sigmoid").trainable = False

        def transfer_filter(paths):
            return [path for path in paths if path.split("_-_")[1] == name]

        train_model(model=model, epochs=args.epochs,
                    model_location=os.path.join(cwd, "output", "model_transfer_{}_{}_{}.h5".format(name, args.window,
                                                                                                   args.band)),
                    log_location=os.path.join(cwd, "output", "training_transfer_{}_{}_{}.log".format(name, args.window,
                                                                                                     args.band)),
                    data_folder=data_folder, batch_size=args.batch, time_window=args.window,
                    batch_equalizer=Default2EnvBatchEqualizer(), filters=[transfer_filter])
