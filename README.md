# AICUP-aortic-valve-detection
This repository is the source code for AI-CUP 2025 Fall Competition owned by **TEAM_8123**.

Our team members:
- 黃佳倫 Chia-Lun Huang
- 劉家民 Jason Liu
- 莊品謙 Pin-Chien Chuang

The specification our hardware is as below
| Hardware     | Specifications |
|:-------------|:---------------|
| OS  | Ubuntu 24.04 LTS|
| GPU | RTX 5090 32GB|
| RAM | 32GB |

> [!NOTE]  
> This code walkthrough is under **Jupyter Notebook** running on **Ubuntu linux**.

The code that can be used is such as:

`main.ipynb`: Main part of the code

`aortic-classify.ipynb`: Filtering dataset code

`other-models.ipynb`: Additional models that can be used/modify for further research


## Environment
**STEP 1**\
First, we need to initiate a python virtual environment, as shown below (Run the command in **bash**):
```bash
python3 -m venv <venv-name>
```
After initialization, activate the virtual environment and install the required packages in `requirements.txt` (Run the command in **bash**).
```bash
source <venv-name>/bin/activate
pip install -r requirements.txt
```

**STEP 2**\
After installation for the additional packages, connect `<venv-name>` into Jupyter Notebook's kernel (Run the command in **bash**).
```bash
python -m ipykernel install --user --name=<venv-name> --display-name "<kernel-name>"
```
`<venv-name>` : initialized python virtual environment

`<kernel-name>` : kernel display name in Jupyter Notebook

## Code Guide
The following section explains about data preparation, training the model, making the prediction, and ensembling the prediction results for different models.

### I. Preparing the Data
**STEP 1**\
Unzip the training and testing dataset (Run the command in **bash**).
```bash
unzip -q training_image.zip -d .
unzip -q training_label.zip -d .
unzip -q testing_image.zip -d .
```

**STEP 2**\
Combine all of the training dataset into one directory.\
Modifyable: `START`, `END`
```
datasets
|---images
|---labels
```

**STEP 3**\
Create a `.yaml` file for training and testing dataset.\
Modifyable: `OUTPUT_ROOT`, `FOLDS`, `CLASS_NAME`
```python
OUTPUT_ROOT = './datasets_kfold/'
CLASS_NAME  = 'aortic_valve'
```

If the dataset directory is similar to the example below, the dataset is ready to be passed on to train the model.
```
datasets_kfold
|---fold_0
|   |---train
|   |   |---images
|   |   |---labels
|   |---val
|   |---dataset_fold_0.yaml
|---fold_1
|      ⋮
|---fold_6
```


### II. Training the Model
Specify the hyperparameter to train the YOLO model.\
Modifyable: `FOLD_NUM`,`EPOCH`,`BATCH`, `MODEL_PATH`, `CLASS_NAME`,`SAVE_PATH`,`DATASET_PATH`,`SKIP_MODEL`
```python
MODEL_PATH    = "yolo12x.pt"
CLASS_NAME    = "aortic_valve"

SAVE_PATH     = f"original_{i}"
DATASET_PATH  = f"./datasets_kfold/fold_{i}/dataset_fold_{i}.yaml"
```
|Training|Time(hr)|
|:-------|:------:|
|Original|1.55    |
|Box15   |2.06    |
|Dropout |2.06    |

The table above shows the approximate time needed to train the model for each fold

### III. Prediction
#### 1. Filtering Dataset through Classification
Modifyable: `IMAGE_ROOT`,`LABEL_ROOT`,`OUTPUT_ROOT`,`CLASS_NAMES`,`MODEL_PATH`,`EPOCHS`,`BATCH`,`FOLDS_NUM`,`RANDOM_SEED`
```python
IMAGE_ROOT  = "./datasets/images"
LABEL_ROOT  = "./datasets/labels"
OUTPUT_ROOT = "./classification_dataset"
MODEL_PATH  = "yolo11n-cls.pt"
```

|Training|Time(min)|
|:---|:---:|
|yolo11n-cls.pt|12.53|

|Classify   |Time(ms)|
|:----------|:------:|
|Preprocess |  0.7   |
|Inference  |  0.3   |
|Postprocess|  0.0   |

**STEP 1**: Prepare dataset\
Separate the training files into 5 folds

**STEP 2**: Training\
Train model on each fold

**STEP 3**: Filtering\
From testing dataset, filter images with very low confidence.\
Modifyable: `MODEL_PATHS`, `TEST_IMAGE_ROOT`, `OUTPUT_ROOT`, `CONFIDENCE_THRESHOLD`, `CLASS_NAMES`

```python
MODEL_PATHS     = ["path/to/cls-modelA.pt", "path/to/cls-modelB.pt", ⋯]
TEST_IMAGE_ROOT = "./datasets/testing"
OUTPUT_ROOT     = "path/to/classified/dataset/"
```

#### 2. Predicting using Filtered Dataset
Specify the classified dataset path, trained model and output `.txt`.\
Modifyable: `FOLD_NUM`,`IOU`,`CONF`,`CLASSIFIED_PATH`,`TRAINED_MODEL_PATH`,`OUTPUT_FILE_NAME`,`SKIP_MODEL`

```python
CLASSIFIED_PATH    = 'path/to/classified/dataset/'
TRAINED_MODEL_PATH = 'path/to/modelA.pt'
OUTPUT_FILE_NAME   = 'predictionA.txt'
```

|Prediction |Time(ms)|
|:----------|:------:|
|Preprocess |  0.9   |
|Inference  | 23.9   |
|Postprocess|  0.3   |

The table above shows the approximate time needed to do inference per image in milliseconds.

### IV. Ensembling Prediction Results
Specify the predicted results `.txt` file path from different models into the variable `prediction_list`, and the ensembled prediction into `output_file`.\
Modifyable: `IMG_WIDTH`, `IMG_HEIGHT`, `prediction_list`, `IOU`, `CONF`,`output_files`

```python
prediction_list = ["path/to/predictionA.txt", "path/to/predictionB.txt", ⋯]
output_file = 'ensembled_prediction_modelA_modelB.txt'
```
