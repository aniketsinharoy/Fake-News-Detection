import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes() #loading dataset of diabetes in a variable

#print(diabetes.keys())
#tell us veryting about the data and other information too
#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

#print(diabetes.DESCR)
# Data Set Characteristics:
#   :Number of Instances: 442
#   :Number of Attributes: First 10 columns are numeric predictive values
#   :Target: Column 11 is a quantitative measure of disease progression one year after baseline
#   :Attribute Information:
#       - age     age in years
#       - sex
#       - bmi     body mass index
#       - bp      average blood pressure
#       - s1      tc, total serum cholesterol
#       - s2      ldl, low-density lipoproteins
#       - s3      hdl, high-density lipoproteins
#       - s4      tch, total cholesterol / HDL
#       - s5      ltg, possibly log of serum triglycerides level
#       - s6      glu, blood sugar level
      
#print(diabetes.feature_names)
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

#print(diabetes)
#print everything that is in diabetes dataset

diabetes_X_single_feature = diabetes.data[:, np.newaxis, 2]
#extracting a single attribute/feature having index 2 date in a new variable
#print(diabetes_X_single_feature)
#NOTE: to extract all feature data use diabetes.data

diabetes_X_train = diabetes_X_single_feature[0:-30]     #taking training data from starting to last 30th element i.e 0 to 70
diabetes_X_test = diabetes_X_single_feature[-30:]       #taking last 30 elements i.e 71 to 100

diabetes_Y_train = diabetes.target[0:-30]               #variable holding the output of the traing data set
diabetes_Y_test = diabetes.target[-30:]                 #variable holding the output of the testing data set

LRmodel = linear_model.LinearRegression()
LRmodel.fit(diabetes_X_train,diabetes_Y_train)      #training the model

diabetes_Y_predict = LRmodel.predict(diabetes_X_test)   #predicting the result of unknown data

print("Mean Square Error: ",mean_squared_error(diabetes_Y_test,diabetes_Y_predict))     #{(p1-o1)^2+(p2-o2)^2+..(pn-on)^2}/n
print("Intercept: ",LRmodel.intercept_)
print("Weight Wo: ",LRmodel.coef_)          #Linear Regression f(x) = WO*Xo + I

#this plot can be when when we are using only single feature
plt.scatter(diabetes_X_train,diabetes_Y_train)  #plot the points only
plt.plot(diabetes_X_test,diabetes_Y_predict)    #plot the points and draw a line between them
plt.show()                                      #show the graph