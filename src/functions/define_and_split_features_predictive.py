#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to define features and split data
def define_and_split_features(sales_train_with_prices):
    features = ['store_id', 'item_id', 'day', 'month', 'weekday', 'sell_price', 'event_name_encoded', 'event_type_encoded']
    X = sales_train_with_prices[features]
    y = sales_train_with_prices['sales']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

