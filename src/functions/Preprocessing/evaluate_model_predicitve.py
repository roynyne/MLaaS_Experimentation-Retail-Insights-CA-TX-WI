#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to evaluate model
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Test RMSE with {model_name}: {rmse}")
    return rmse

