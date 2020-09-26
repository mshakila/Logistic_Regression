#### LOGISTIC REGRESSION ASSIGNMENT - BANK dataset

# Business Problem : To classify if customer will subscribe or not to a term deposit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

# EDA
bank_raw = pd.read_csv("D:\\bank-full.csv",header=0,sep=';')
bank_raw.columns
''' 'age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'y' 
numeric var: age, balance, day,duration,campaign, pdays, previous
categorical var: 'job', 'marital', 'education', 'default', 'housing',
       'loan', 'contact', 'month','poutcome', 
dependent variable: 'y' - whether taken TD (Term Deposit) or not      
       '''
bank_raw.head()
bank_raw.describe()
bank_raw.dtypes
 
bank_raw.y.value_counts()
''' no     39922
   yes     5289 '''

sns.countplot(x='y', data=bank_raw)

y_no = len(bank_raw[bank_raw['y']=='no'])
y_yes = len(bank_raw[bank_raw['y']=='yes'])
y_no/(y_no+y_yes) #  88.3%
y_yes/(y_no+y_yes)    # 11.69%
# one class has very low proportion, this is an imbalanced dataset

bank_raw.groupby('y').mean()
''' the customers having more age have taken TD.
Customers who have maintained more balance in their accounts have taken TD
Customers who were contacted lesser number of times in this campaign have taken TD.'''

bank_raw.groupby('job').mean()
pd.crosstab(bank_raw.job, bank_raw.y).plot(kind='bar')
''' based on job type, we can know if TD is taken or not. For example, persons 
in management cadre, blue collar jobs, technical jobs, admin and services should be
given more importance in the campaigns.'''

bank_raw.age.hist();plt.xlabel('Age');plt.ylabel('Frequency');
# most of the bank customers age is between 26 to 40 years (for this dataset)


# To get the count of null values in the data 

bank_raw.isnull().sum() # no missing values

########## creating dummy columns for the categorical columns 
#  categorical var: 'job', 'marital', 'education', 'default', 'housing',
#  'loan', 'contact', 'month','poutcome'

bank_dummies = pd.get_dummies(bank_raw[['job', 'marital', 'education', 'default', 'housing','loan', 'contact', 'month','poutcome']])
# 44 dummy varaibles have been created out of 9 categorical variables

# Dropping the columns for which we have created dummies
# copying bank_raw in a new dataset 
bank_raw1 = bank_raw.copy()
bank_raw1.drop(['job', 'marital', 'education', 'default', 'housing','loan', 'contact', 'month','poutcome'], inplace=True, axis=1)

# adding the dummy variables to the bank_raw1 data frame 
bank = pd.concat([bank_raw1, bank_dummies],axis=1)

bank.columns
# 7th column is 'y'
bank.iloc[:6,7]

# splitting dataset
from sklearn.model_selection import train_test_split
train, test = train_test_split(bank,test_size=0.3, random_state=123)

train_X = train.loc[:,train.columns != 'y']
train_y = train.loc[:,train.columns == 'y']
test_X = test.loc[:, test.columns != 'y']
test_y = test.loc[:, test.columns == 'y']

train_y.y.value_counts()
3723/31647
test_y.y.value_counts()
1566/13564
# percentage of y is 12% in train and test dataset. 
# If we were to predict all as 'no', even then we will get 82% accuracy.

''' This is an imbalanced dataset. The results of logistic regression will get biased 
towards majority class (no TD). So before model building we have to balance the dataset. '''

# lets build model on this imbalanced dataset.
# train the model
logreg_imbalanced = LogisticRegression().fit(train_X, train_y)
pred = logreg_imbalanced.predict(test_X)
pred = pd.DataFrame(pred)

confusion_matrix(pred,test_y)
#      [[11706,  1059],
#      [  292,   507]]
11706/13564
# Here only if we consider correctly predicted TN (11706), accuracy is 86%
accuracy_score(pred,test_y) # 90% 
precision_score(pred,test_y,pos_label='yes') # 0.3238 out of predicted positives, 32% are correctly predicted
recall_score(pred,test_y,pos_label='yes') # 0.6345 out of actual postives, 63% are correctly predicted by the model
f1_score(pred,test_y,pos_label='yes') # 0.4288 this balances precision and recall

######################## OVERSAMPLING of minority class
# we have already split data into train and test.  let us use train data for oversampling

# now separate minority and majority classes
no_TD = train[train.y == 'no']
yes_TD = train[train.y =='yes']
 
# oversampling the minority class
from sklearn.utils import resample

bank_over = resample(yes_TD, replace=True, n_samples=len(no_TD),random_state=123)

# combining majority and oversampled minority
oversampled = pd.concat([no_TD, bank_over])

# checking new class counts
oversampled.y.value_counts()
# yes    27924
# no     27924

######################## Model building using oversampled data
X_train_over = oversampled.drop('y',axis=1)
y_train_over = oversampled.y

logreg_oversampled = LogisticRegression().fit(X_train_over,y_train_over)

pred_over = logreg_oversampled.predict(test_X)

# accuracy measurements
confusion_matrix(pred_over,test_y)
# 10189,   296
# 1809,  1270
accuracy_score(pred_over,test_y) # 0.8448 previously 90%
precision_score(pred_over,test_y,pos_label='yes') # 0.8109 out of predicted positives, 81% are correctly predicted
# Now it is 81% (previous model just 32%)
recall_score(pred_over,test_y,pos_label='yes') # 0.4125 out of actual postives, 41% are correctly predicted by the model
# Now it is 41% (previous model 63%)
f1_score(pred_over,test_y,pos_label='yes') # 0.5468 
# Now it is 55% (previous model 43%)

''' Though accuracy has decreased, f1_score has increased. Also the models prediction ability
has increased.'''

###################### Generating Synthetic samples using SMOTE

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

sm = SMOTE(random_state=123, ratio=1.0)
X_train_smote, y_train_smote = sm.fit_sample(train_X,train_y)
# 55848 obs in each 

classifier = LogisticRegression()
logreg_smote = LogisticRegression().fit(X_train_smote,y_train_smote)

pred_smote = logreg_smote.predict(test_X)

# accuracy measurements
confusion_matrix(pred_smote,test_y)
# 10713, 614
# 1285,  952
accuracy_score(pred_smote,test_y) # 0.86 
precision_score(pred_smote,test_y,pos_label='yes') # 0.6079 
recall_score(pred_smote,test_y,pos_label='yes') # 0.4256 
f1_score(pred_smote,test_y,pos_label='yes') # 0.5007 


'''
            logreg_imbalanced    logreg_oversampled   logreg_smote
accuracy          0.90                 0.84               0.86
precision         0.32                 0.81               0.61 
recall            0.63                 0.41               0.43  
f1                0.43                 0.55               0.50

Accuracy and recall scores are better in model with imbalanced data
Precision and f1 scores are better in oversampled model
Overall, oversampled model is better.

CONCLUSIONS

We have to classify customers those who have subscribed to Term deposits from 
those who have not. Since the output variable is categorical (yes, no), we have 
used Logistic Regression technique.

From exploratory data analysis, we have found that the dataset is highly imbalanced.
So we have tried sampling techniques: oversamping and synthetic sampling. 
We have split the data into train and test: on train data we have done sampling.

We have run 3 different models: one with imbalanced data and others using oversampling
and SMOTE. The model built using oversampling technique is better. 

Due to time limitation we have tried only 3 models. We can build models using 
undersampling, tree-based algorithms, Cost-based algorithm, change the split ratio 
for train-test, change random_state, etc. Then choose the best model.
 
'''



















