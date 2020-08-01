import glob
import os
import tensorflow as tf
from loguru import logger


class Default2EnvParser(object):
    """Default parser for the EEG with matched and mismatched envelope paradigm"""

    def __init__(self):
        self.window_size = None

    def __call__(self, serialized_data):
        """Serialize the EEG, matched envelope and mismatched envelop data

        Parameters
        ----------
        serialized_data : tf.Tensor
            Serialized protobuf data from .tfrecords

        Returns
        -------
        tuple(tf.Tensor, tf.Tensor, tf.Tensor)
            Tensors of the EEG, matched envelope and mismatched envelope data
        """
        data = tf.io.parse_example(serialized_data, {
            'eeg': tf.io.FixedLenFeature((64,), tf.float32),
            'good_env': tf.io.FixedLenFeature((1,), tf.float32),
            'bad_env': tf.io.FixedLenFeature((1,), tf.float32),
        })
        return tf.reshape(data["eeg"], shape=(-1, 64)), tf.reshape(data["good_env"], shape=(
            -1, 1)), tf.reshape(data["bad_env"], shape=(-1, 1))


class TFRecordsDatasetBuilder(object):
    """Create a tf.data.Dataset object from our collection of .tfrecords"""
    separator = "_-_"

    def __init__(self, folder, filters=tuple()):
        """Initialize the TFRecordsDatasetCreator

        Parameters
        ----------
        folder : str
            Path to the folder
        filters : list(Callable)
            List of callable filter functions to filter which data to use
        """
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.filters = filters

    def subjects(self, set_name):
        """All subjects currently selected for the provided set_name"""
        return list(set([x.split(self.separator)[1] for x in self.file_mapping[set_name]]))

    def stimuli(self, set_name):
        """All stimuli currently selected for the provided set_name"""
        return list(set([x.split(self.separator)[-1].split(".tfrecords")[0] for x in self.file_mapping[set_name]]))

    @property
    def file_mapping(self):
        """Mapping between files and sets"""
        return {"train": self.train_files, "validation": self.validation_files, "test": self.test_files}

    @property
    def train_files(self):
        return self._apply_filters(glob.glob(os.path.join(self.folder, "train*.tfrecords")))

    @property
    def validation_files(self):
        return self._apply_filters(glob.glob(os.path.join(self.folder, "validation*.tfrecords")))

    @property
    def test_files(self):
        return self._apply_filters(glob.glob(os.path.join(self.folder, "test*.tfrecords")))

    def _apply_filters(self, paths):
        """Apply filters to paths"""
        for filter in self.filters:
            paths = filter(paths)
        return paths

    def prepare(self, set_name, window=640, window_overlap=0.9, batch_size=1, batch_equalizer=None,
                parser=Default2EnvParser(), cutoff=None):
        """Prepare a dataset for a specific set

        Parameters
        ----------
        set_name : {"train", "validation", "test"}
        window : int
            Window data by this amount
        window_overlap : float
            Overlap between the windows
        batch_size : int
            Batch size this dataset will generate
        batch_equalizer : BatchEqualizer
            Object to make sure that labels in batches are balanced
        parser : Default2EnvParser
            Parser to de-serialize the data in .tfrecords protobufs

        Returns
        -------
        tf.data.Dataset or list(tf.data.Dataset)
            Dataset for the provided set_name. If set_name == test, then a dictionary with a subject-code:dataset mapping will be provided
        """
        logger.info("Loading %u files for %s" % (len(self.file_mapping[set_name]), set_name))
        if set_name == "test":
            datasets = {}
            for subject in self.subjects(set_name):
                datasets[subject] = self._prepare(
                    [x for x in self.file_mapping[set_name] if subject == x.split(self.separator)[1]], window=window,
                    window_overlap=window_overlap, batch_size=batch_size,
                    batch_equalizer=batch_equalizer, parser=parser, cutoff=cutoff
                )
            return datasets
        else:
            return self._prepare(
                self.file_mapping[set_name], window=window, window_overlap=window_overlap, batch_size=batch_size,
                batch_equalizer=batch_equalizer, parser=parser, cutoff=cutoff
            )

    def _prepare(self, file_paths, window=640, window_overlap=0.9, batch_size=1, batch_equalizer=None,
                 parser=Default2EnvParser(), cutoff=None):
        """Prepare a dataset for a specific set

        Parameters
        ----------
        file_paths : list(str)
            Paths to the files
        window : int
            Window data by this amount
        window_overlap : float
            Overlap between the windows
        batch_size : int
            Batch size this dataset will generate
        batch_equalizer : BatchEqualizer
            Object to make sure that labels in batches are balanced
        parser : Default2EnvParser
            Parser to de-serialize the data in .tfrecords protobufs
        Returns
        -------
        tf.data.Dataset
            The interleaved dataset for specified file_paths
        """
        dataset = tf.data.Dataset.from_tensor_slices(file_paths).interleave(
            lambda x: tf.data.TFRecordDataset(x, buffer_size=tf.compat.v1.py_func(os.path.getsize, [x], tf.int64))
                .window(window, shift=int((1 - window_overlap) * window), drop_remainder=True)
                .flat_map(lambda x: x.batch(window)),
            cycle_length=8, block_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        if batch_equalizer is not None:
            dataset = dataset.map(batch_equalizer, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset


class BatchEqualizer(object):

    def __init__(self):
        self.window_size = None
        self.batch_size = None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_window_size(self, window_size):
        self.window_size = window_size

    def __call__(self, *args, **kwargs):
        pass


class Default2EnvBatchEqualizer(BatchEqualizer):
    """Class to make sure the labels of the are balanced"""
    def __init__(self, nb_label_outputs=1):
        """Initialize

        Parameters
        ----------
        nb_label_outputs : int
            How many outputs should be provided. Only usefull if you want to create models in parallel
            to increase GPU usage
        """
        super().__init__()
        self.prepend_size = None
        self.nb_label_outputs = nb_label_outputs

    def __call__(self, eeg, good_env, bad_env):
        """Equalize the labels.
        This will double the batch size

        Parameters
        ----------
        eeg : tf.Tensor
            The EEG data
        good_env : tf.Tensor
            The matched envelope data
        bad_env : tf.Tensor
            The mismatched envelope data

        Returns
        -------
        tuple(tuple(tf.Tensor, tf.Tensor, tf.Tensor), tuple(tf.Tensor,...))
            A tuple formatted according to Keras fit parameters. First element is the inputs to the model,
            second element is the ouputs (labels) for this model. If nb_label_outputs > 1,
            then there will be duplicates of this output
        """
        new_eeg = tf.concat([eeg, eeg], axis=0)
        env1 = tf.concat([good_env, bad_env], axis=0)
        env2 = tf.concat([bad_env, good_env], axis=0)
        labels = tf.concat([
            tf.tile(tf.constant([[1]]), [tf.shape(eeg)[0], 1]),
            tf.tile(tf.constant([[0]]), [tf.shape(eeg)[0], 1]),
        ], axis=0)
        all_labels = []
        for x in range(self.nb_label_outputs):
            all_labels += [labels]
        return (new_eeg, env1, env2), tuple(all_labels)


def sort_dict_items_by_train(d, train_name="train"):
    """Get the items of d sorted by the train_name"""
    return sorted(list(d.items()), key=lambda s: s[0] == train_name,
                  reverse=True)


def test_dataset(ds):
    it = ds.make_one_shot_iterator()
    i = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            # Keep running next_batch till the Dataset is exhausted
            while True:
                i += 1
                a = sess.run(it.get_next())
                if hasattr(a, "shape"):
                    a = a.shape
                elif isinstance(a, tuple):
                    new_a = []
                    for x in a:
                        if hasattr(x, "shape"):
                            new_a += [x.shape]
                        else:
                            new_a += ["length: %u" % len(x)]
                    a = new_a
                logger.info(("batch %u" % i) + str(a))

        except tf.errors.OutOfRangeError:
            pass


subjects = ["2019_C2DNN_1", "2019_C2DNN_2", "2019_C2DNN_3", "2019_C2DNN_4", "2019_C2DNN_5", "2019_C2DNN_6",
            "2019_C2DNN_7", "2019_C2DNN_9", "2019_C2DNN_10", "2019_C2DNN_11", "2019_C2DNN_12", "2019_C2DNN_13",
            "2019_C2DNN_14", "2019_C2DNN_15", "2019_C2DNN_16", "2019_C2DNN_17", "2019_C2DNN_18", "2019_C2DNN_20",
            "2019_C2DNN_21", "2019_C2DNN_23", "2019_C2DNN_24", "2019_C2DNN_25", "2019_C2DNN_28", "2019_C2DNN_29",
            "2019_C2DNN_30", "A05S09", "B30K04", "E03C02", "E07S07", "E16D09", "E26L12", "G12A10", "J09H12", "J27V07",
            "L02J10", "L16W05", "L27G06", "L30P03", "M12C10", "M25G07", "M27J04", "P27T02", "S06L11", "L07N05",
            "S14S04", "S16T09", "S23T01", "V09L10"]
