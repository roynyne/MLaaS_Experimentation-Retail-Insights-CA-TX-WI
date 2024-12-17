Sales Revenue Prediction and Forecasting
==============================

Experiments pertaining to the development of forecasting and predictive models for the purpose of analysing and projecting sales income for a retail organisation having locations in Wisconsin (WI), Texas (TX), and California (CA) are included in this repository. The models will be made available as production-ready APIs.

### Project Overview

The project aims to: ○ 

Assignment 2

ML as a Service: 
Sales Revenue Prediction and Time-Series Forecasting for a Multi-State Retailer

Roy Hegde
Student ID: 24667610
Github Username
roynyne
Github Repos
Experiment Repository:
https://github.com/roynyne/36120_AT2_experimentations 
API Repository:
https://github.com/roynyne/sales-api 
URLs
Backend: https://sales-api-backend-qw48.onrender.com 
Frontend: https://sales-api-frontend.onrender.com 


      
       
       36120 - Advanced Machine Learning Application
Master of Data Science and Innovation
University of Technology of Sydney
Table of Contents

1. Executive Summary	2
2. Business Understanding	4
a. Business Use Cases	4
3. Data Understanding	6
4. Data Preparation	7
5. Modeling	8
a. Approach 1: Ridge Regression	8
b. Approach 2: XGBoost	9
c. Approach 3: ARIMA (Forecasting Model)	9
d. Approach 4: Prophet (Forecasting Model)	10
6. Evaluation	11
a. Evaluation Metrics	11
b. Results and Analysis	11
c. Business Impact and Benefits	13
d. Data Privacy and Ethical Concerns	13
7. Deployment	14
Backend Deployment with FastAPI	14
Frontend Deployment with Streamlit	16
Simultaneous Deployment on Render	20
8. Conclusion	21

# 1. Executive Summary

The goal of this project is to create machine learning models for a store that has locations in Wisconsin, Texas, and California. Predicting item-specific sales income at the shop level and projecting the overall sales for all stores over the following seven days were the objectives. The project's goal was to give the shop useful information for improving labour scheduling, marketing tactics, and inventory control.

Figure 1.1 Retail Store

Two models were developed:
○ Predictive Model: Sales revenue for a certain item at a given store on a given day was predicted using XGBoost.
○ Forecasting Model: For the following seven days, total sales across all stores were predicted using the time-series forecasting algorithm ARIMA.

The shop was able to obtain real-time sales forecasts when both models were made available as Render APIs. With Streamlit on the front end and FastAPI on the back end, the deployment was made easier and the user experience was smooth. Because the models provide precise forecasts which are essential for labour allocation and inventory management—the company can save operating expenses. For the predictive test, XGBoost obtained a Root Mean Squared Error (RMSE) of 3.013 whereas ARIMA outperformed Prophet in terms of dependable performance for time-series forecasting.


# 2. Business Understanding

## a. Business Use Cases
Keeping up with changing demand may be difficult for retail firms when it comes to personnel, marketing, and inventory management. This project's retailer, which has ten locations in three states, needs precise sales projections to handle the following:

○ Inventory control: While stockouts result in lost sales and disgruntled customers, overstocking can raise holding expenses. To effectively manage inventory, the store has to be able to forecast demand for certain goods.
○ Labour Planning: Inaccurate estimates might result in stores having too few employees during busy times or too many employees during calm times, which would be inefficient use of labour.
○ Marketing and Promotions: Targeting promotions and running more efficient marketing campaigns are two ways to increase profitability when you know when and where demand will jump. to speak to:

In order to assist managers make data-driven choices, the machine learning models created for this project provide forecasts that specifically meet these business demands. The shop can enhance promotional activities, effectively schedule workers, and optimise inventory levels using precise demand estimates.

## b. Key Objectives

The project has three main objectives:
○ Create a Predictive Model: Based on past sales data, event information, and price, machine learning algorithms are used to anticipate sales income for certain goods at individual stores.
○ Create a Model for Forecasting: To anticipate the entire sales income for all retailers over the next seven days, use a time-series forecasting model. Allocating resources and strategic planning will be aided by this paradigm.
○ Models to be Deployed as APIs: In order to facilitate real-time engagement, both models will be implemented as APIs, enabling stakeholders to obtain forecasts whenever needed.

Stakeholders include:
○ Store managers are in charge of making sure that personnel and inventory levels correspond to demand.
○ The operations team allocates resources throughout retailers and manages supply chain logistics using projections.
○ Marketing Department: Based on anticipated demand, forecasts are used to optimise promotional efforts and modify strategy.

By addressing these key objectives, the project empowers the retailer to enhance its operational efficiency and improve decision-making processes.

# 2. Data Understanding

The study made use of many datasets from different sources, each of which provided crucial data for model training:

○ Sales Training Data: This dataset includes daily sales information for certain products sold in a number of different retailers. It serves as the main dataset for the prediction model and offers a detailed historical record of sales patterns. Each row includes the item ID, shop ID, and many identifiers such as department and category, and it also represents the daily sales of an item at a certain store.
○ Calendar Data: Weekends, holidays, and other noteworthy events that can have an impact on sales trends are among the crucial date-related details that are captured in the calendar data. Seasonality and trends, which are critical for time-series forecasting models like ARIMA and Prophet, may be found in this data.
○ Event Data: Promotions and vacations, for example, are known to have a big influence on customer behaviour. The event dataset contains details on a number of annual occasions, such Black Friday and Thanksgiving, that might have an impact on sales. The model may take these demand surges into account by integrating this data.
○ Weekly Sales Data for goods: This dataset compiles pricing and sales information for individual goods on a weekly basis, giving users a general idea of how prices change over time. Given the importance of price in consumer decision-making, this information is critical to the predictive and forecasting models.

Every dataset has some restrictions. For instance, several item prices in the sales data were missing, and some holidays' event data was lacking. During the data preparation stage, these problems were resolved via preprocessing and data cleaning.

Figure 3.1 Sales Training Data

Figure 3.2 Calender Data

Figure 3.3 Calender Events Data

Figure 3.4 Items Weekly Sales Data


# 3. Data Preparation

A number of crucial procedures were engaged in data preparation to guarantee that the datasets were appropriate for model training:

○ Dataset Combination: 
    ○ Reshaping Sales Data: The melt function was used to convert the sales data from wide to long format. This made it possible to show the sales of each item on a given day for            each row.
    ○ Merging Calendar Data: Using the day reference (d), the reshaped sales data was combined with calendar data to provide attributes relating to time and date.
    ○ Incorporating Event Information: To account for the influence of promotions and holidays, calendar event data on the date column was combined to create special events.
    ○ Adding Price Data: In order to account for price variations that effect sales, item prices were finally merged depending on wm_yr_wk, item_id, and store_id as well.

○ Data cleaning: Forward-filling techniques were used to fill in the missing data, particularly in the item price columns. The model was given complete training data by filling          up the missing variables using historical prices. To enable the algorithm to discern between genuine events and ordinary sales periods, entries without event data were                 labelled as "No Event."

○ Feature Engineering: In order to better capture the underlying patterns in the data, new features were produced from old ones via feature engineering, which was a crucial step in      the data preparation process:
    ○ Date Features: To help the model understand how time affects sales patterns, the "date" column was divided into many aspects, such as the day, month, and weekday.
    ○ Label Encoding: Label encoding was applied to categorical variables, such as store_id, item_id, event_name, and event_type. This made it possible for machine learning                  algorithms to handle these category characteristics numerically, such as XGBoost and Ridge.
        
○ Managing Outliers and Missing Data: To maintain the integrity of the dataset, outliers in the sales data were retained, while forward-fill methods were utilised to fill in the         missing values in non-essential categories. Price data consistency was guaranteed by the forward-fill approach, and outliers offered vital information about atypical sales spikes—     a critical component for models attempting to understand the impact of uncommon occurrences.
    
○ Data Splitting: Training and testing sets of the data were created. Twenty percent of the data were set aside for testing and eighty percent were utilised for training the             prediction model. To make sure the forecasting model was not trained on any future data, time-based validation was employed, whereby data from previous periods was utilised to         predict future patterns.


Figure 4.1 Calender Data


Figure 4.2 Encoded Event Name Values


Figure 4.3 Encoded Event Type Values


# 4. Modeling

## a. Approach 1: Ridge Regression
The baseline model for item-specific sales prediction was Ridge Regression, a linear model that supplements ordinary least squares with L2 regularisation. Ridge is especially helpful in situations when the data exhibits multicollinearity, or strongly linked properties. Due to its ability to penalise high coefficients and hence avoid overfitting, the regularisation is a viable contender for initial model selection.
Why Ridge Regression?
Ridge Regression is a great place to start for predictive modelling because of its simplicity, particularly in retail settings where the relationship between sales and characteristics (such item price, shop location, or promotions) might not seem to be linear at first.

The dataset underwent preprocessing to provide significant characteristics prior to the fitting of the Ridge Regression model:

○ Date Features: Day, month, and weekday variables were created from the 'date' field. Retail sales are cyclical and seasonal, as seen by the fact that they usually rise on weekends     and fall in the middle of the week.
○ Event Encoding: Event_name_encoded and Event_type_encoded are categorical characteristics that were label-encoded using holiday and event data, which is essential for retail           forecasting. The model was able to take into consideration surges during important sales occasions like Black Friday or Christmas thanks to this modification.
○ Store and Item Encoding: To give numeric values to category store and item IDs, store_id and item_id were both label-encoded. To capture trends unique to individual items in           stores, these aspects are crucial.

## b. Approach 2: XGBoost
The second strategy was XGBoost (Extreme Gradient Boosting), a potent and well-liked ensemble learning technique that excels at managing big datasets, feature interactions, and non-linear correlations. XGBoost's performance and versatility have made it a leading algorithm for structured/tabular data certain patterns.

○ Similar to Ridge Regression, XGBoost's performance was greatly enhanced via feature engineering:
○ Price Normalisation: To assist the model train more efficiently, the sell_price feature was normalised to standardise the range of prices across various goods.
○ Event Features: To capture the effect of holidays and special events on sales, two features were used: event_name_encoded and event_type_encoded. These characteristics enabled XGBoost to modify its forecasts according to the presence of promotions or the holiday season on a given day.
○ Date data: To better capture cyclical sales trends, XGBoost benefited from extra date data such as the week of the year and whether the day was a weekday or a weekend.


Figure 5.b.1  XGB Model Parameters

## c. Approach 3: ARIMA (Forecasting Model)
ARIMA (Auto-Regressive Integrated Moving Average) was chosen as the main model for time-series forecasting. The retailer's sales data included both seasonality and observable patterns, which make this approach especially useful.

### Model Selection and Tuning

Three critical parameters of the ARIMA model need to be carefully adjusted: p (number of lag observations), d (degree of differencing), and q (moving average window size). The model that performed the best after experimenting with several parameter combinations was ARIMA(5, 1, 0). These specifications showed shown in Figure 5.c.1:
○ p=5: The model predicts the subsequent value based on five prior observations.
○ d=1: To make the series stationary and account for trends over time, a single differencing was used.
○ q=0: Since the moving average window wasn't needed for the sales data, it wasn't applied in this instance.


Figure 5.c.1  ARIMA Parameters

## c. Approach 4: Prophet (Forecasting Model)
From Figure 6.2, prophet's ability to manage complicated or erratic sales patterns brought on by holidays and promotions is one of its distinctive qualities. Users may enter particular holidays or events that could affect sales into Prophet, and the program modifies its forecast appropriately. In this project, Prophet was fed special event data, including Thanksgiving and other holidays, so it could account for the corresponding surges in sales.
Though Prophet could identify these anomalous sales trends, its performance fell short of ARIMA in this particular instance. The retailer's sales data showed high weekly seasonality, which ARIMA handled more skilfully with simple differencing and autoregressive terms. This might be one explanation for the observed behaviour.

# 6. Evaluation

## a. Evaluation Metrics
The main assessment statistic for the forecasting and predicting models was RMSE, or root mean squared error. The average discrepancy between the expected and actual sales numbers is measured by RMSE. It was especially crucial in this situation since even little fluctuations in sales forecasts can have a big impact on the retailer's operations, resulting in things like excess inventory or understaffing.

## b. Results and Analysis
○ Ridge Regression: This baseline model has an RMSE of 3.544.
○ XGBoost: With an RMSE of 3.013, this prediction model was chosen as the final one because of its excellent results.
○ ARIMA: Selected as the ultimate forecasting model due to its precise predictions and dependable national sales forecasting performance.


Figure 6.1 XGB & Ridge Regression Results


Figure 6.2 fbprophet Results


Figure 6.3 ARIMA Results


## c. Business Impact and Benefits

The models created have significant business value.
○ XGBoost: The merchant may more effectively manage inventory and steer clear of stock-related problems by precisely projecting sales for individual goods at certain shops. As a result, there are lower overstocking expenses and fewer stockouts that result in missed revenues.
○ ARIMA: The store can best allocate labour and manage resources by using the national sales prediction. The retailer can staff stores adequately and make sure supply chain operations are in line with predicted demand by having knowledge of future sales patterns.

The models not only increase operational efficiency but also give the marketing team insightful information. The retailer may more efficiently arrange promotions and boost sales and customer happiness by pinpointing times of strong demand.

## c. Data Privacy and Ethical Concerns
Anonymised data was utilised in this project to guarantee that no private client information was revealed. The models' primary focus on sales patterns and outside events reduced the likelihood of data privacy violations. To guarantee that all shops and areas were treated equally in the projections, care was taken to prevent bias in the models, especially in the event forecasting aspects.

## 7. Deployment

Two essential elements had to be put up in order for the predictive and forecasting models to be deployed:

○ FastAPI Backend: For managing forecast and prediction queries.
○ Streamlit Frontend: To offer an interactive user interface for accessing the models. 

To provide scalability and accessibility, both services were deployed on Render after being containerised using Docker. An outline of the deployment structure may be seen below, along with thorough descriptions of each part.

### Backend Deployment with FastAPI
FastAPI, a cutting-edge, asynchronous web framework perfect for managing real-time forecasts and predictions, was used in the development of the backend API. There are two primary uses for the API:

Forecast sales by item with the XGBoost model.
Using the ARIMA model, provide a seven-day national sales estimate.

The main.py file defines the API endpoints:
○ Root Endpoint: Shows instructions on how to use the API and a welcome message.
○ Healthcheck Endpoint: Attests to the models operational readiness.
○ Prediction endpoint (/sales/stores/items/): Returns sales projections based on store ID, item ID, date, price, and event information at the.
○ Forecasting Endpoint (/sales/national/): Based on the commencement date, offers a seven-day projection for national sales.


Figure 7.1 GET (“/”) root welcome message

Figure 7.2 GET ("/health") health check

CORS (Cross-Origin Resource Sharing) was enabled in order to guarantee smooth communication between the frontend and backend services, enabling requests from the Streamlit interface hosted on Render.

Figure 7.3 Bridging both ends with CORS

### Docker for Backend
Docker was used to containerise the FastAPI backend, and the Dockerfile specified the environment and required libraries to operate the API. The backend may be readily installed on Render or other cloud services, and it is accessible via port 8000.


Figure 7.4 FastAPI/Backend Docker File

### Frontend Deployment with Streamlit

Streamlit was used in the frontend's construction to provide an interactive user experience.

Enter the pricing, event details using encoded integers from Figure 4.2 & Figure 4.3, store and item details, and predict store/item sales. The FastAPI backend receives these inputs from Streamlit and returns the anticipated sales.
With the start date supplied by the user, forecast national sales over the next seven days. The FastAPI endpoint receives the request from the frontend and provides the predicted values in a table.

Figure 7.5 Streamlit Interface Input Screen

Figure 7.6 ("/sales/stores/items/") Results

Figure 7.7 ("/sales/national/") Results

### Docker for Frontend
Docker is used to containerise and deploy the Streamlit frontend, much like it is for the backend. The frontend uses HTTP requests to establish direct communication with the FastAPI backend while operating on port 8501.


Figure 7.8 Streamlit/Frontend Docker File

### Simultaneous Deployment on Render

The frontend (Streamlit) and backend (FastAPI) were both deployed on Render, guaranteeing smooth communication between the two services. With the frontend acting as the user interface and the backend performing model inference, Render's free tier makes it simple to set up and host both services.

Render Links:
○ Backend API: https://sales-api-backend-qw48.onrender.com 
○ Frontend App: https://sales-api-frontend.onrender.com


Figure 7.9 Render Web Services for Streamlit & FastAPI


# 8. Conclusion
Through the development and implementation of machine learning models for sales forecasting and prediction, this project effectively met its goals. Because of its accuracy and performance, the XGBoost model was picked for item-store-date forecasts, whereas the ARIMA model was chosen for time-series forecasting across all stores. The store may make better decisions about labour allocation, marketing tactics, and inventory management with the help of these actionable data from both models.
To further improve forecast accuracy, future studies may involve adding other factors, such as weather data. Maximising the advantages of these models might also include expanding the solution's reach to more locations and connecting it with the retailer's larger supply chain management system.



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
    ├── reports              
    │   └── figures          
    │   └── 36120_AdvMLA_AT2_Roy_Hegde24667610.pdf          <- Final Report in pdf format.
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
