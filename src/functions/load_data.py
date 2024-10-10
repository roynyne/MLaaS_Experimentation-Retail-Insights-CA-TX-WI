#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from prophet import Prophet
import matplotlib.pyplot as plt
import joblib


# In[ ]:


# Function to load datasets
def load_data():
    sales_train_df = pd.read_csv('/Users/bananavodka/Projects/at2_mla/at2_mla/data/raw/sales_train.csv')
    calendar_df = pd.read_csv('/Users/bananavodka/Projects/at2_mla/at2_mla/data/raw/calendar.csv')
    calendar_events_df = pd.read_csv('/Users/bananavodka/Projects/at2_mla/at2_mla/data/raw/calendar_events.csv')
    sales_test_df = pd.read_csv('/Users/bananavodka/Projects/at2_mla/at2_mla/data/raw/sales_test.csv')
    items_weekly_sales_df = pd.read_csv('/Users/bananavodka/Projects/at2_mla/at2_mla/data/raw/items_weekly_sell_prices.csv')
    
    print("Datasets loaded successfully.")
    return sales_train_df, calendar_df, calendar_events_df, sales_test_df, items_weekly_sales_df

