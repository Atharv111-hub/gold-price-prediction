# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:19:42 2024

@author: prachet
"""

import numpy as np
import pickle
import os  # ✅ Added to handle file paths

# 🔧 Correct way to load the model using absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'gold_price_prediction_model.sav')
loaded_model = pickle.load(open(model_path, 'rb'))

# 💡 Creating a function for prediction
def gold_price_prediction(input_data):
    # changing the input data to numpy
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshape the array as we are predicting on 1 instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction
