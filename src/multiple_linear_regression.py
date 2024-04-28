# import libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Import data and split dependent and independent variables
dataset = pd.read_csv("data/50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Imput missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, :2])
x[:, :2] = imputer.transform(x[:, :2])


# One Hot Encode categorical independent variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough"
)
x = np.array(ct.fit_transform(x))

# Split train and test data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
)
