import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score

def get_data(file_name):
    data = pd.read_csv(file_name)
    data = data.fillna(method='ffill')
    x_parameter = []
    y_parameter = []
    for single_square_feet ,single_price_value in zip(data['Year'],data['MIN(Rs)']):
        x_parameter.append([float(single_square_feet)])
        y_parameter.append(float(single_price_value))
    return x_parameter,y_parameter

x,y = get_data('dataset1.csv')
print(x)
print(y)

dt = tree.DecisionTreeClassifier()
dt.fit(x, y)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(x)
y=(np.array([y]).reshape(-1,1))
Y = sc_Y.fit_transform(y)

#y_rbf=sc_Y.inverse_transform(.predict(sc_X.transform(np.array([2016]).reshape(-1,1))))

predictions = dt.predict(sc_X.transform(np.array([2019]).reshape(-1,1)))
print("Minimum crop price in 2019 predicted by decision tree algorithm")
print(predictions)

plt.scatter(x,y,color='blue')
plt.plot(x,dt.predict(x),color='red',linewidth=4)
plt.xlabel('Years')
plt.ylabel('Crop production')
plt.title('Decision tree')
plt.show()
