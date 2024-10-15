Sales Revenue Prediction and Forecasting
==============================

Experiments pertaining to the development of forecasting and predictive models for the purpose of analysing and projecting sales income for a retail organisation having locations in Wisconsin (WI), Texas (TX), and California (CA) are included in this repository. The models will be made available as production-ready APIs.

### Project Overview

The project aims to:

Create a prediction model to forecast sales income for a certain item in a particular retailer on a given date by utilising machine learning methods (XGBoost, Ridge).
Create a forecasting model to project the total sales income for all retailers over the following seven days using time-series analysis methods (ARIMA, Prophet).

### Repository Structure
------------
    
    ├── LICENSE              <- License file specifying the project’s licensing terms.
    ├── Makefile             <- Makefile with commands like `make data` or `make train` to automate workflows.
    ├── README.md            <- The top-level README file for developers using this project.
    ├── data
    │   ├── external         <- Data from third-party sources such as calendar and event data.
    │   ├── interim          <- Intermediate data that has been cleaned or transformed for further analysis.
    │   ├── processed        <- The final, canonical data sets prepared for modeling.
    │   └── raw              <- The original, immutable data dump, such as sales_train.csv, sales_test.csv, etc.
    │       ├── calendar.csv
    │       ├── calendar_events.csv
    │       ├── items_weekly_sell_prices.csv
    │       ├── sales_test.csv
    │       ├── sales_train.csv
    │       └── sales_train_with_prices1.csv
    │
    ├── docs                 <- Documentation files, potentially built using Sphinx or other documentation tools.
    │
    ├── models               <- Trained and serialized model artifacts, model predictions, or model summaries.
    │   ├── forecasting
    │   │   ├── ARIMA_model.pkl      <- Serialized ARIMA model used for time-series forecasting.
    │   │   └── prophet_model.pkl    <- Serialized Prophet model used for time-series forecasting.
    │   └── predictive
    │       └── xgboost_model.pkl    <- Serialized XGBoost model used for sales revenue prediction.
    │
    ├── notebooks            <- Jupyter notebooks. Naming convention includes creator’s initials, student ID, and description.
    │   ├── forecasting
    │   │   ├── Hegde_Roy-24667610-forecasting_ARIMApipeline.ipynb   <- ARIMA forecasting pipeline.
    │   │   └── Hegde_Roy-24667610-forecasting_fbprophetpipeline.ipynb <- Prophet forecasting pipeline.
    │   └── predictive
    │       └── Hegde_Roy-24667610-predictive_ridgexgb.ipynb         <- Ridge and XGBoost predictive model pipeline.
    │
    ├── references           <- Data dictionaries, manuals, and other reference materials related to the project.
    │
    ├── reports              <- Generated reports in docx
    │   └── figures          <- Generated visualizations and figures used in reporting.
    │
    ├── requirements.txt     <- The requirements file listing all dependencies for the project.
    │                         Used to recreate the project’s environment using `pip install -r requirements.txt`.
    │
    ├── setup.py             <- Makes the project pip-installable (i.e., `pip install -e .`), allowing modules in `src/` to be imported.
    │
    ├── src                  <- Source code for this project, organized in submodules for easy access and reusability.
    │   ├── __init__.py      <- Marks the `src` directory as a Python package.
    │   │
    │   ├── data             <- Scripts for loading or generating the datasets.
    │   │   └── make_dataset.py <- Script for downloading or preprocessing raw data into clean datasets.
    │   │
    │   ├── features         <- Scripts for transforming raw data into features for machine learning models.
    │   │   └── build_features.py  <- Script for feature engineering and building new features.
    │   │
    │   ├── models           <- Scripts for training and using machine learning models.
    │   │   ├── predict_model.py   <- Script for loading models and making predictions.
    │   │   └── train_model.py     <- Script for training models on datasets.
    │   │
    │   └── visualization    <- Scripts for generating visualizations.
    │       └── visualize.py  <- Script for creating plots and visualizing the data or model results.
    │
    └── tox.ini              <- Configuration file for Tox, a tool for automating testing environments.

--------

### Installation
------------

1. Clone the repository:

    git clone <repo_link>
    cd sales-revenue-prediction

2. Install the required packages:

    pip install -r requirements.txt

--------

### Notebooks
------------

The main notebooks can be found in the /notebooks/ folder, structured as follows:

**Predictive Models:**

Notebooks for Ridge and XGBoost models.
1. Located in /notebooks/predictive/.
2. Example: Hegde_Roy-24667610-predictive_ridgexgb.ipynb.
--------

**Forecasting Models:**
------------

Notebooks for ARIMA and Prophet models.
1. Located in /notebooks/forecasting/.
2. Example: Hegde_Roy-24667610-forecasting_ARIMApipeline.ipynb.
--------

### Custom Modules
------------

The custom modules used for data processing, feature engineering, and model training are stored in the /src/ directory. These include:

Model Functions (src/functions/)

    ├── functions
    │   ├── ARIMA           <- Functions to run an ARIMA model.
    │   │   ├── forecast_arima.py
    │   │   ├── main_arima.py
    │   │   └── train_arima.py
    │   ├── Preprocessing           <- Set of functions to load dependencies and preprocessing.
    │   │   ├── create_features.py
    │   │   ├── define_and_split_features_predictive.py
    │   │   ├── evaluate_model_predicitve.py
    │   │   ├── load_data.py
    │   │   └── preprocess_data.py
    │   ├── XGB_Ridge           <- Functions to run both XGB and Ridge Regression model.
    │   │   ├── main_xgb_ridge.py
    │   │   ├── train_ridge.py
    │   │   └── train_xgboost.py
    │   └── prophet           <- Functions to run a Prophet model.
    │       ├── aggregate_sales_forecasting.py
    │       ├── main_forecasting.py
    │       ├── make_forecast.py
    │       └── train_prophet.py

--------

### Models
------------

Trained models are stored in the /models/ folder:

Predictive Models (/models/predictive/): XGBoost model artifact.
Forecasting Models (/models/forecasting/): ARIMA and Prophet models.

--------

### Running the Code
------------

**Run Jupyter Notebooks:**

    Use Jupyter Lab/Notebook to run the notebooks in /notebooks/ for training and evaluating the models.

    Example Command :

    poetry shell
    jupyter notebook

**Loading the Trained Models:**

    The order of the functions can be referred through /notebooks/ for both prediction and forecasting

    The trained models are available in the /models/ folder and can be loaded directly using joblib or pickle.
    use "main_<models>" in the end to run as input for all functions.
--------


### Contributors
------------

Roy Hegde (roynyne@hotmail.com)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
