import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

def get_data(file_name):
    data = pd.read_csv(file_name)
    x_parameter = []
    y_parameter = []
    for single_square_feet ,single_price_value in zip(data['Year'],data['Annual Rainfall']):
        x_parameter.append([float(single_square_feet)])
        y_parameter.append(float(single_price_value))
    return x_parameter,y_parameter

x,y = get_data('dataset1.csv')
print(x)
print(y)

lr = LinearRegression()
lr.fit(x, y)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(x)
y=(np.array([y]).reshape(-1,1))
Y = sc_Y.fit_transform(y)

#y_rbf=sc_Y.inverse_transform(.predict(sc_X.transform(np.array([2016]).reshape(-1,1))))

predictions = lr.predict(sc_X.transform(np.array([2019]).reshape(-1,1)))
print("Annual Rainfall for 2019 predicted by Linear Regression")
print(predictions)

plt.scatter(x,y,color='blue')
plt.plot(x,lr.predict(x),color='red',linewidth=4)
plt.xlabel('Years')
plt.ylabel('Annual Rainfall(mm)')
plt.title('Linear Regression')
plt.show()
