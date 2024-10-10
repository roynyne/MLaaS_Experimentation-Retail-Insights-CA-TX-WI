#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to train XGBoost model
def train_xgboost(X_train, y_train):
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    print("XGBoost model trained successfully.")
    return xgb_model

