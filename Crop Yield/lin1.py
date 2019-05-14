import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

dataTrain = pd.read_excel("dataset1.xlsx")

x_train = dataTrain[['Year', 'Arrival']].values.reshape(-1,2)
y_train = dataTrain['MAX']

regr = linear_model.LinearRegression()
model = regr.fit(x_train, y_train)
x_test=[2015, 15879]
print model.predict(x_test)


