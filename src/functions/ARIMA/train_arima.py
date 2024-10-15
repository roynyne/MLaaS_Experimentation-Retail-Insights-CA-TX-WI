#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to train ARIMA model
def train_arima(sales_data):
    # Convert the aggregated sales data into a time series
    sales_series = sales_data.set_index('ds')['y']
    
    # Train ARIMA model (ARIMA(5, 1, 0) as an example, modify parameters as needed)
    arima_model = ARIMA(sales_series, order=(5, 1, 0))
    arima_result = arima_model.fit()
    
    # Save the ARIMA model using joblib
    #joblib.dump(arima_result, '/Users/bananavodka/Projects/at2_mla/at2_mla/models/forecasting/ARIMA_model.pkl')
    #print("ARIMA model saved successfully.")
    
    return arima_result

