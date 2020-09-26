# Logistic Regression - affairs dataset

# Business Problem: To classify persons having affairs and those not having any affairs


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# EDA
affairs_raw = pd.read_csv("D:\\affairs.csv")
affairs_raw.columns
affairs_raw.head()
affairs_raw.describe()
affairs_raw.dtypes

affairsdata = affairs_raw.copy()
affairsdata['affairs']=(affairsdata.affairs >0).astype(int)
affairsdata.affairs.value_counts()
# 0 451 and  1 150

affairsdata.groupby('affairs').mean()
''' there is very less difference in variables in terms of mean. those who are not
having affairs have given higher rating for marriage (more happier), former are
more religious
'''

# visualizations
sns.countplot(x="affairs",data=affairsdata)
# the proportion of thos not having affairs is more than that of having affairs
sns.countplot(x="gender",data=affairsdata)
# our dataset has a few more females than males
sns.countplot(x="children",data=affairsdata)
# lesser persons in the dataset have no children as compared to those havng children
affairsdata.religiousness.hist()

# barplot of religiousness grouped by affair
pd.crosstab(affairsdata.religiousness, affairsdata.affairs).plot(kind='bar')
# barplot of religiousness grouped by affair
pd.crosstab(affairsdata.religiousness, affairsdata.affairs).plot(kind='bar')
# barplot of yearsmarried grouped by affair
pd.crosstab(affairsdata.yearsmarried, affairsdata.affairs).plot(kind='bar')

# finding missing values in the data 

affairsdata.isnull().sum() # no missing values

# creating dummy variables for categorical varaibles
affairs_dummies = pd.get_dummies(affairsdata[["gender","children"]])
# Dropping the columns for which we have created dummies
affairsdata.drop(["gender","children"],inplace=True,axis = 1)

# adding the columns to the affairsdata data frame 

affairsdata = pd.concat([affairsdata,affairs_dummies],axis=1)
##################### Model building

affairsdata.shape # 601 records ,11 variables
X = affairsdata.iloc[:,[1,2,3,4,5,6,7,8,9,10]]
Y = affairsdata.iloc[:,0]
classifier = LogisticRegression()
classifier.fit(X,Y)

classifier.coef_ # coefficients of features 
classifier.predict_proba (X) # Probability values 

y_pred = classifier.predict(X)
affairsdata["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df = pd.concat([affairsdata,y_prob],axis=1)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)

# accuracy 
from sklearn import metrics
classifier.score(X,Y) # 76%
(437+20)/601
affairsdata.affairs.value_counts()
'''
here actuals are in the rows,  though accuracy is 76% , True postive is very less 20
our of 150.'''

# evaluate model using 10-fold cross validatation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LogisticRegression(),X,Y,scoring='accuracy', cv=10)
print(scores)
# [0.72131148 0.73333333 0.73333333 0.75       0.76666667 0.75
# 0.76666667 0.78333333 0.73333333 0.76666667]
scores.mean() # 75%

#the accuracy is same as when we used the complete dataset to train and predict 

'''
CONCLUSIONS

we are classifying persons those who are having affairs and those who are not
having affairs. We are using logistic regression to do the binary calssification.

we have built a standard model using complete dataset. We have evaluated model
by predicting on the whole dataset and also used 10-fold cross validation.

The accuracies obtained were same in both. We can improve the accuracy by using other
classification techniques like Naive Bayes, KNN and also by increasing the sample size.
'''



