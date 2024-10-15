#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to make future predictions and plot the forecast using Prophet
def make_forecast(prophet_model, periods=7):
    # Create a dataframe for the next `periods` days
    future_dates = prophet_model.make_future_dataframe(periods=periods)

    # Make predictions
    forecast = prophet_model.predict(future_dates)

    # Plot the forecast using Prophet's built-in plot method
    fig = prophet_model.plot(forecast)
    
    # Show the plot
    plt.show()

    # Return the forecasted values
    print("Forecast generated and plotted successfully.")
    return forecast

