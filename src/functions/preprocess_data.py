#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to preprocess and merge datasets
def preprocess_data(sales_train_df, calendar_df, calendar_events_df, items_weekly_sales_df):
    # Melt sales data
    sales_train_melted = pd.melt(
        sales_train_df,
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        var_name='d',
        value_name='sales'
    )
    
    # Merge with calendar data
    sales_train_merged = pd.merge(sales_train_melted, calendar_df, how='left', on='d')

    # Merge with event data
    sales_train_with_events = pd.merge(sales_train_merged, calendar_events_df, how='left', on='date')

    # Merge with item prices
    sales_train_with_prices = pd.merge(sales_train_with_events, items_weekly_sales_df, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])

    # Fill missing event names and types
    sales_train_with_prices['event_name'].fillna('No Event', inplace=True)
    sales_train_with_prices['event_type'].fillna('None', inplace=True)

    # Forward-fill missing prices
    sales_train_with_prices['sell_price'].fillna(method='ffill', inplace=True)
    sales_train_with_prices['sell_price'].fillna(
        sales_train_with_prices.groupby('item_id')['sell_price'].transform('mean'), 
        inplace=True
    )

    print("Data preprocessed successfully.")
    return sales_train_with_prices

