import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("data/Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_gen = PolynomialFeatures(degree=4)
x_poly = poly_gen.fit_transform(x)
regressor = LinearRegression()
regressor.fit(x_poly, y)

fig, axes = plt.subplots()
axes.scatter(x, y, color="red")
axes.plot(x, regressor.predict(x_poly), color="blue")
plt.show()

print("predicted f(6.5):")

print(regressor.predict(poly_gen.fit_transform([[6.5]])))
