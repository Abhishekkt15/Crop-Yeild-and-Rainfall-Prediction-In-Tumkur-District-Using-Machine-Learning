import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model 

def get_data(file_name):
    data = pd.read_excel(file_name)
    x_parameter = []
    y_parameter = []
    for single_square_feet ,single_price_value in zip(data['years'],data['annuallrainfall']):
        x_parameter.append([float(single_square_feet)])
        y_parameter.append(float(single_price_value))
    return x_parameter,y_parameter

x,y = get_data('tr.xlsx')
print x
print y

def linear_model_main(X_parameters,Y_parameters,predict_value):
 
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    predict_outcome = regr.predict(predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions

#x,y = get_data('input_data.csv')
predict_value = 2016
result = linear_model_main(x,y,predict_value)
print "Intercept value " , result['intercept']
print "coefficient" , result['coefficient']
print "Predicted value: ",result['predicted_value']

# Function to show the resutls of linear fit model
def show_linear_line(X_parameters,Y_parameters):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    plt.scatter(X_parameters,Y_parameters,color='blue')
    plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)
    #plt.axis([ 2003, 2015, 140, 1200])
    plt.xlabel('Years')
    plt.ylabel('Annuall Rainfall(mm)')
    plt.title('Tumkur District Annual Rainfall')
    #plt.axis([2003, 2015, 400, 1200])
    #plt.xticks(())
    #plt.yticks(())
    plt.grid(True)
    plt.show()

 	
show_linear_line(x,y)
