#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Main function to execute the entire pipeline
def main():
    # Load data
    sales_train_df, calendar_df, calendar_events_df, sales_test_df, items_weekly_sales_df = load_data()

    # Preprocess data
    sales_train_with_prices = preprocess_data(sales_train_df, calendar_df, calendar_events_df, items_weekly_sales_df)

    # Create features
    sales_train_with_prices, encoder_store, encoder_item, le_event_name, le_event_type = create_features(sales_train_with_prices)

    # Define and split features
    X_train, X_test, y_train, y_test = define_and_split_features(sales_train_with_prices)

    # Train Ridge model
    ridge_model = train_ridge(X_train, y_train)

    # Evaluate Ridge model
    evaluate_model(ridge_model, X_test, y_test, model_name="Ridge Regression")
    
    # Train XGBoost model
    X_train_subset = X_train.sample(frac=0.10, random_state=42)
    y_train_subset = y_train[X_train_subset.index]
    xgb_model = train_xgboost(X_train_subset, y_train_subset)

    # Evaluate XGBoost model
    evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")

if __name__ == "__main__":
    main()

