import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

dataTrain = pd.read_csv("dataset1.csv")

x_train = dataTrain[['Year']].values.reshape(-1,1)
y_train = dataTrain['Annual Rainfall']





import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
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



reg = linear_model.LinearRegression()
reg.fit(x, y)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
x = sc_X.fit_transform(x)
y=(np.array([y]).reshape(-1,1))
x = sc_Y.fit_transform(y)

print(reg.predict(sc_X.transform(np.array([2016]).reshape(-1,1))))

plt.scatter(x,y,color='blue')
plt.plot(x,reg.predict(x),color='red',linewidth=4)
plt.xlabel('Years')
plt.ylabel('Annuall Rainfall(mm)')
plt.title('Tumkur District Annual Rainfall')
plt.show()
