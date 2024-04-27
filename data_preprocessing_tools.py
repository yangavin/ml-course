# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import dataset and separate independent and dependent variables
dataset = pd.read_csv("data/Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Impute missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# One Hot Encode categorical independent variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough"
)
x = np.array(ct.fit_transform(x))

# Label encode dependent variable
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

# Splitting training set and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print("x_train: ", x_train)
print("x_test: ", x_test)
print("y_train: ", y_train)
print("y_test: ", y_test)
