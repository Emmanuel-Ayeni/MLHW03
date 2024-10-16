# MLHW03
Bank Marketing Campaign - End-to-End Machine Learning Project. This project demonstrates an end-to-end machine learning pipeline applied to the publicly available Bank Marketing dataset from the UCI Machine Learning Repository. 

The dataset consists of data related to direct marketing campaigns of a Portuguese banking institution, the task is to predict whether a client will subscribe to a term deposit based on various features like age, job, marital status, and contact methods.

### Project Overview
The project covers all stages of a typical machine learning workflow:

Data Understanding and Preprocessing:
Load and inspect the dataset.
Handle missing values, encoding categorical features, and feature scaling.
Exploratory Data Analysis (EDA):
Visualise data distributions, and correlations, and identify key trends.

### Feature Engineering:
Create new features to improve model performance.
Deal with imbalanced data.

### Modeling:
Train various machine learning models (e.g., Logistic Regression, Random Forest, XGBoost).
Tune hyperparameters using GridSearchCV.

### Model Evaluation:
Evaluate models using accuracy, precision, recall, F1-score, and AUC-ROC curve.

### Deployment:
Export the best-performing model for deployment using joblib.
Optional: Creating a simple Flask API for model inference.

### Dataset
The dataset can be found on the UCI Machine Learning Repository:
Bank Marketing Dataset

### Features
Input Variables: Information about the client, bank, and marketing campaign.
Output Variable: Whether the client subscribed to a term deposit (yes or no).

### Project Steps
Data Loading: The raw data is loaded from the .zip file available from the UCI repository.
Data Cleaning & Preprocessing: The data is cleaned, including handling missing data, encoding categorical variables, and scaling numerical features.

#### Exploratory Data Analysis
Exploratory Data Analysis (EDA): A series of visualizations and statistical analyses are performed to uncover relationships in the data.

#### Model
Modelling: Several machine learning models are trained and evaluated using cross-validation.

#### Hyperparameter Tuning
Hyperparameter Tuning: Hyperparameters are fine-tuned using GridSearchCV to find the best model.

#### Model Evaluation
Model Evaluation: The models are evaluated based on key metrics such as accuracy, F1-score, and AUC-ROC.

#### Deployment
Deployment: The model is exported for deployment, with a basic API interface example included.

### How to Run the Project
#### Clone the repository:
bash
git clone https://github.com//bank-marketing-ml.git

#### Install dependencies:
bash
pip install -r requirements.txt

Run the Jupyter Notebook to explore the data, train the models, and evaluate performance.

To deploy the model:
Optional, use the Flask API for inference (see the api.py file for details).
The model can be loaded using joblib for deployment in your application.


### Results
The project evaluates various machine learning models to predict whether a client will subscribe to a term deposit. The best model is selected based on performance metrics like AUC-ROC, precision, and recall.

### Technologies Used
Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
Jupyter Notebooks
Flask (for optional model deployment)
Joblib (for model saving and loading)

### Acknowledgments
The dataset provided by the UCI Machine Learning Repository: Bank Marketing Dataset.
