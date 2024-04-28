import pandas as pd
import matplotlib.pyplot as plt

# Importing and split dataset
dataset = pd.read_csv("data/salary_data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split test and train set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train simple linear regression model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict
y_pred = regressor.predict(x_test)

# Plot
plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.show()
