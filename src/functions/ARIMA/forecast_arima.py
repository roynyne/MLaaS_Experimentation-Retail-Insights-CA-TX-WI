#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to forecast using ARIMA model
def forecast_arima(arima_model, periods=7):
    forecast = arima_model.forecast(steps=periods)
    
    # Create future dates for the forecast
    last_date = arima_model.data.dates[-1]
    future_dates = pd.date_range(last_date, periods=periods+1, freq='D')[1:]
    
    # Plot the forecast
    plt.figure(figsize=(10,6))
    plt.plot(forecast.index, forecast, label='Forecast')
    plt.title("ARIMA Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()
    
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Sales': forecast})
    print("ARIMA Forecast generated successfully.")
    return forecast_df

