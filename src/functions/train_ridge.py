#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function to train Ridge model
def train_ridge(X_train, y_train):
    ridge_model = Ridge()
    ridge_model.fit(X_train, y_train)
    print("Ridge model trained successfully.")
    return ridge_model

