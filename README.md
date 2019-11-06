Structure of this toolkit
===

To get you started and to make sure that code will be a bit organized, we propose following structure.

```
thesis_template/
├── custom_code
│   ├── data
│   │   └── dataset_builder.py
│   └── keras
│       └── callbacks.py
├── dataset
│   └── starting_data
├── experiments
│   └── baseline_model
│       ├── evaluation.py
│       ├── experiment.py
│       ├── images
│       ├── model.py
│       ├── output
│       │   ├── best_model.h5
│       │   ├── eval.json
│       │   └── training.log
│       └── visualization.ipynb
├── extra_reports
├── .gitignore
├── README.md
└── requirements.txt
```
In `custom_code`, you can put your more general purpose code, used by different experiments.
In the `custom_code/data/dataset_builder.py` file, you can find code to create datasets.


In the `dataset` directory, the datasets are stored. The `start_dataset` folder contains all `.tfrecords` for the starting dataset.


In the experiment folder, all your experimentation will be contained.
For each experiment, at least 3 files should be present: 
`experiment.py`, `model.py` and `visuzalization.py` as well as one folder: `output`.
`model.py` will contain the code to construct your model.
If `experiment.py` is run, your model should be trained. 
Make sure to always save your model in the `output` folder (preferably as `model.h5`).
If `evaluation.py` is run, your model will be evaluated. 
The results of this evaluation (accuracy, loss,...) should be saved in the `output` folder (preferably as `evaluation.json`).
In  `visualization.ipynb` you can create a nice report of your experimentation with some cool visualizations.

In the `extra_reports` folder, you can put additional jupyter notebooks, comparing results between experiments


To get you started, a `requirements.txt` file is also provided. 


Programming
===
You can use this file to install dependencies with pip like this:
```
pip3 install -r requirements.txt
```
or with conda:
```
conda install --file requirements.txt
```

Be aware that this toolkit was tested with Python 3.6 and uses the `tensorflow` framework.
We suggest using the `keras` API of tensorflow, which is easy-to-use.

**You will need to install tensorflow-gpu separately if you want to train your models on the gpu, it is not included in the requirements.txt file**

Data structure
---
The data we provide are stored in .tfrecords.

Data was collected by letting normal hearing people listen to natural running Flemish speech while recording their EEG.
They listened to 8 (fairytale) stories of 15 minutes. These 8 stories are not available for every subject, due too malfunction of recording/preprocessing.

We split each recording in 3 sets: a training (80% of the data), validation (10% of the data) and test set (10% of the data).
The test and validation set are both extracted from the middle of the recording, as it is unprobable for edge-effects to occur there.

Each `.tfrecords` file has the same naming scheme:
```
{set_name}_-_{subject-code}_-_{story}.tfrecords

e.g.
train_-_2019_C2DNN_22_-_MILAN.tfrecords
```

Every tf.Example (~=sample) in the tfrecords has 3 features:
* `eeg`: A sample of EEG data
* `good_env`: A sample of time-aligned (=matched) stimulus envelope
* `bad_env`: A sample of stimulus envelope taken 11s in the future (=mis-matched)

## Preprocessing

This data is already preprocessed and normalized. The preprocessing steps are:

* For the EEG:
    * Downsampling to 1Khz
    * Artefact rejection with a multi channel Wiener filter (removing eyeblinks)
    * Filtering between [0.5 - 32]Hz
    * Downsampling to 64Hz
    * Standardizing per recording (subtracting mean, dividing by std)
    
* For the Envelope
    * Envelope estimation with gammatone filterbank
    * Downsampling to 1Khz
    * Artefact rejection with a multi channel Wiener filter (removing eyeblinks)
    * Filtering between [0.5 - 32]Hz
    * Downsampling to 64Hz
    * Standardizing per recording (subtracting mean, dividing by std)
