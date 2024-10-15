#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Main function to execute the entire pipeline
def main():
    # Load datasets
    sales_train_df, calendar_df, calendar_events_df, sales_test_df, items_weekly_sales_df = load_data()

    # Preprocess data
    sales_train_with_prices = preprocess_data(sales_train_df, calendar_df, calendar_events_df, items_weekly_sales_df)

    # Create features
    sales_train_with_prices, encoder_store, encoder_item, le_event_name, le_event_type = create_features(sales_train_with_prices)

    # Aggregate daily sales for Prophet
    sales_data = aggregate_sales(sales_train_with_prices)

    # Train Prophet model
    prophet_model = train_prophet(sales_data)

    # Make forecast for the next 7 days
    forecast = make_forecast(prophet_model, periods=7)

    # Display forecasted values
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7))

if __name__ == "__main__":
    main()

