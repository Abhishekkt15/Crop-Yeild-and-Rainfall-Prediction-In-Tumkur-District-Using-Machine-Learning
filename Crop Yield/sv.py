import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR

def get_data(file_name):
    data = pd.read_csv(file_name)
    x_parameter = []
    y_parameter = []
    for single_square_feet ,single_price_value in zip(data['Year'],data['Annual Rainfall']):
        x_parameter.append([float(single_square_feet)])
        y_parameter.append(float(single_price_value))
    return x_parameter,y_parameter

X,y = get_data('dataset1.csv')
print(X)		
print(y)




'''from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='linear') 
  
# fitting x samples and y classes 
clf.fit(x, y) 

clf.predict([[120, 990]]) 
  
clf.predict([[85, 550]]) '''

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=1)
svr_rbf.fit(X, y)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
y=(np.array([y]).reshape(-1,1))
Y = sc_Y.fit_transform(y)

y_rbf=sc_Y.inverse_transform(svr_rbf.predict(sc_X.transform(np.array([2016]).reshape(-1,1))))

#y_rbf=svr_rbf.predict(2016) 
svr_lin.fit(X, y)
y_lin = svr_lin.predict(sc_X.transform(np.array([2016]).reshape(-1,1)))

#y_poly = svr_poly.fit(X, y).predict(2016)

print("2016 rainfall prediction done by Rbf")
#print(y_rbf)
print("2016 rainfall prediction done by linear_model")
print(y_lin)

lw = 2




plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X,svr_lin.predict(X),color='red',linewidth=4)
#plt.hold('on')
#plt.plot(X, svr_rbf.predict(X), color='navy', lw=lw, label='RBF model')
#plt.plot(X, svr_lin.predict(X), color='red', lw=lw, label='Linear model')
#plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('Year')
plt.ylabel('Annual Rainfall(mm)')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
