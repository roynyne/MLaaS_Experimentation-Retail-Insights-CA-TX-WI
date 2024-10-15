#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def main():
    # Load datasets
    sales_train_df, calendar_df, calendar_events_df, sales_test_df, items_weekly_sales_df = load_data()

    # Preprocess data
    sales_train_with_prices = preprocess_data(sales_train_df, calendar_df, calendar_events_df, items_weekly_sales_df)

    # Create features
    # Only capture the first value (the DataFrame) from the create_features function
    sales_train_with_prices, _, _, _, _ = create_features(sales_train_with_prices)

    # Aggregate daily sales for Prophet/ARIMA
    sales_data = aggregate_sales(sales_train_with_prices)

    # Train ARIMA model
    arima_model = train_arima(sales_data)

    # Forecast using ARIMA for the next 7 days
    arima_forecast = forecast_arima(arima_model, periods=7)

    # Display forecasted values
    print(arima_forecast)

if __name__ == "__main__":
    main()

