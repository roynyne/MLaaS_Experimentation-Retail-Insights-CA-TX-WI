#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to aggregate daily sales across all stores and items for Prophet
def aggregate_sales(sales_train_with_prices):
    # Aggregate total daily sales
    sales_data = sales_train_with_prices.groupby('date').agg({'sales': 'sum'}).reset_index()

    # Rename columns to match Prophet's expected format
    sales_data.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
    
    print("Sales data aggregated successfully.")
    return sales_data

