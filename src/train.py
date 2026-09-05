import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("D:\\MlOps Day 1\\data\\data.csv")



# Train-test split
X,y = df[["TV","Radio", "Newspaper"]],df[["Sales"]]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=67
)


# Create model
model = LinearRegression()


# Train model
model.fit(Xtrain, ytrain)


# Prediction
ypred = model.predict(Xtest)


# Evaluation
rmse = root_mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)


print("RMSE:", rmse)
print("R2 Score:", r2)


# Save model
joblib.dump(model, "model/linear_reg_model.pkl")


print("Model saved successfully!")