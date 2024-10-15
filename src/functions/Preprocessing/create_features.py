#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to create features
def create_features(sales_train_with_prices):
    # Convert 'date' to datetime
    sales_train_with_prices['date'] = pd.to_datetime(sales_train_with_prices['date'])

    # Create date-based features
    sales_train_with_prices['day'] = sales_train_with_prices['date'].dt.day
    sales_train_with_prices['month'] = sales_train_with_prices['date'].dt.month
    sales_train_with_prices['weekday'] = sales_train_with_prices['date'].dt.weekday

    # Label encoding
    encoder_store = LabelEncoder()
    encoder_item = LabelEncoder()
    sales_train_with_prices['store_id'] = encoder_store.fit_transform(sales_train_with_prices['store_id'])
    sales_train_with_prices['item_id'] = encoder_item.fit_transform(sales_train_with_prices['item_id'])

    # Initialize label encoders for event names and types
    le_event_name = LabelEncoder()
    le_event_type = LabelEncoder()
    sales_train_with_prices['event_name_encoded'] = le_event_name.fit_transform(sales_train_with_prices['event_name'])
    sales_train_with_prices['event_type_encoded'] = le_event_type.fit_transform(sales_train_with_prices['event_type'])

    print("Features created successfully.")
    return sales_train_with_prices, encoder_store, encoder_item, le_event_name, le_event_type

