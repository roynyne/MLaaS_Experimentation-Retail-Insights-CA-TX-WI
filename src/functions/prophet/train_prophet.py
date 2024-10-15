#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to train Prophet model
def train_prophet(sales_data):
    # Initialize Prophet model
    prophet_model = Prophet()

    # Fit the model on the aggregated sales data
    prophet_model.fit(sales_data)

    print("Prophet model trained successfully.")
    return prophet_model

