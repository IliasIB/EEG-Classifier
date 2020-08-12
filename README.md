# Match/Mismatch Prediction Between Speech Stimuli and EEG Using Deep Transfer Learning
This repository is an implementation of the thesis Match/Mismatch Prediction Between Speech Stimuli and EEG Using Deep Transfer Learning.

## Structure
This project is structured in the following way:
* **custom_code** contains helper functions used for the experiments.
* **experiments** contains the different experiments that were performed, which are the baseline_model, recurrent_model and transfer_model. All experiments have the following files that can be used:
  * **experiment.py** can be run with the respective parameters to execute the experiment and thus train the models (use -h to get a list of the parameters required)
  * **evaluation.py** can be run with the respective parameters to execute the evaluation of the model trained during the experiment (use -h to get a list of the parameters required)
  * **visualization.ipynb** is a notebook that contains the visualizations made from the experiments that were done (this should only be run after executing the necessary experiments and evaluations with the correct parameters)

## Installation
You can use this file to install dependencies with pip like this:
```
pip3 install -r requirements.txt
```
or with conda:
```
conda install --file requirements.txt
```

Be aware that this project was tested with Python 3.6 and uses the `tensorflow` framework with version 1.14. A guide to installing tensorflow can be found at https://www.tensorflow.org/install. Take care to install the correct version.

Before running any experiments make sure your PYTHONPATH is set correctly to the root:
```
export PYTHONPATH=/path/to/project:$PYTHONPATH
```

## Weights
Although the dataset cannot be provided, the weights for all experiments can be downloaded from https://www.dropbox.com/s/h5ct9tpylkp2n4z/weights.zip?dl=0. Simply extract the contents of the zip file to the root of the folder.
