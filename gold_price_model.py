# gold_price_model.py

import numpy as np
import pickle
import os

# Load the model from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'gold_price_prediction_model.sav')
loaded_model = pickle.load(open(model_path, 'rb'))

def gold_price_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction
