import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

dataTrain = pd.read_csv("dataset1.csv")
#dataTest = pd.read_csv("dataTest.csv")
# print df.head()

x_train = dataTrain[['Year', 'Annual Rainfall', 'Area']].values.reshape(-1,3)
y_train = dataTrain['crop_Production']

#x_test = dataTest[['Temperature(K)', 'Pressure(ATM)']].reshape(-1,2)
#y_test = dataTest['CompressibilityFactor(Z)']

ols = linear_model.LinearRegression()
model = ols.fit(x_train, y_train)
x_test = [2016, 511, 34589] 
#print model.predict(x_train)
print(model.predict(x_test))

"""plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,model.predict(x_train),color='red',linewidth=4)
plt.xlabel('Years')
plt.ylabel('Annuall Rainfall(mm)')
plt.title('Tumkur District Annual Rainfall')
plt.show()"""
