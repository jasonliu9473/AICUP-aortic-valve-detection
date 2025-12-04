# AICUP-aortic-valve-detection
This repository is the source code for AI-CUP 2025 Fall Competition owned by **TEAM_8123**.

Our team members:
- 黃佳倫 Chia-Lun Huang
- 劉家民 Jason Liu
- 莊品謙 Pin-Chien Chuang

## Environment
### Step 1
First, we need to initiate a python virtual environment, as shown below:
```
pip -m venv <venv-name>
```
After initialization, install the required packages in `requirement.txt`.
### Step 2
After installation for the additional packages, connect `<venv-name>` into Jupyter Notebook's kernel.
```
python -m ipykernel install --user --name=<venv-name> --display-name "<kernel-name>"
```
`<venv-name>` : initialized python virtual environment

`<kernel-name>` : kernel display name in Jupyter Notebook

## Code Guide
The following section explains about data preparation, training the model, making the prediction, and ensembling the prediction results for different models.

### Preparing the Data


### Training the Model


### Prediction
#### 1. Classifying Testing Dataset
> [!NOTE]
> This step can be skipped, we use classification model in order to reduce outlier prediction.


#### 2. Using the Prediction Model


### Ensembling Prediction Results
Specify the predicted results `.txt` file path from different models into the variable `prediction_list`, and the ensembled prediction into `output_file`

```python
prediction_list = ["path/to/predictionA.txt", "path/to/predictionB.txt", ⋯]
output_file = "ensembled_prediction_modelA_modelB.txt"
```
