import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(filepath_or_buffer='./data/irisdataset.csv')

#Selecting Independent(i/p or features) and Dependent(o/p or target) variables
X = df.iloc[:, :4]
y = df.iloc[:,-1]
#Splitting Dataset Into Training & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Scaling Independent Data
# Scaling helps to bring the dataset into the range of -1 to 1. 
# It is not mandatory but it is always good to scale our dataset before processing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training Model Using Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_predict = lr.predict(X_test_scaled)

# check the accuracy of logistic regression model
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_predict, y_test)
print(f'The Accuracy score of logistic regression over dataset:', acc_score)

#storing model as pickle 
import pickle
pickle.dump(lr, open('./models/models.pkl', 'wb'))

# Using model to make predictions
model = pickle.load(open('./models/models.pkl', 'rb'))
test_df = pd.DataFrame({
    'Sepal_Length': [5.1],
    'Sepal_Width': [5.1],
    'Petal_Length': [5.2],
    'Petal_Width': [3.2]
})
print('Test DataFrame', test_df)
test_data = scaler.transform(test_df)
print(test_data)
print(lr.predict(test_data))



