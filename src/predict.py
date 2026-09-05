import joblib
import pandas as pd
import numpy as np


new_data = pd.DataFrame([[100, 250, 198]], columns=["TV", "Radio", "Newspaper"])

model = joblib.load("D:\MlOps Day 1\model\linear_reg_model.pkl")

predictions = model.predict(new_data)

print(f"Predicted sales is {predictions}") 