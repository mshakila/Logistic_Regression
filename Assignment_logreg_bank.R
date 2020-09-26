# Logistic Regression Assignment on bank dataset

# Business Problem: To classify if customer will opt for a term deposit or not

# Data collection: 
bank_raw <- read.csv("D:\\bank-full.csv", sep = ";")
# 45211 records of 17 variables
names(bank_raw)
'''  [1] "age"       "job"       "marital"   "education" "default"   "balance"  
 [7] "housing"   "loan"      "contact"   "day"       "month"     "duration" 
[13] "campaign"  "pdays"     "previous"  "poutcome"  "y" '''

head(bank_raw)
str(bank_raw)
attach(bank_raw)
table(job) #12 levesls
table(marital) #3 levesls
table(education) #4 levesls
table(default  ) #2 levesls
table(housing  ) #2 levesls
table(loan     ) #2 levesls
table(contact  ) #3 levesls
table(month    ) #12 levesls
table(poutcome     ) #4 levesls

# dependent variable
table(y     ) #2 levesls
prop.table(table(y))
# no-39922  88.3% ,    yes- 5289  11.70%
# our classes here are imbalanced. 

# to know how R dummifies categorical variable
contrasts(bank_raw$y)
# no is 0 and yes is 1 (i.e., not taking TD is taken as 0)

# checking for missing values
sum(is.na(bank_raw))
library(Amelia)
missmap(bank_raw,main='Observed vs missing values')
# the whole plot is blue (for observed values), no missing values

######splitting the dataset
library(caTools)
set.seed(123)
split_data = sample.split(bank_raw$y,SplitRatio=0.6)
train = subset(bank_raw, split_data==T)
test <- subset(bank_raw, split_data==FALSE)

# lets check the proportions of yes and no of y variable
prop.table(table(train$y)) # no is 0.883 and yes is 0.1169
prop.table(table(test$y))  # no is 0.8829 and yes is 0.117
prop.table(table(bank_raw$y))   # no is 0.883 and yes is 0.1169
# the proportions of both classes is same in train, test and original datasets

# we have seen that the data is imbalanced, lets build a model using above 
# random sampling and see the accuracy of this imbalanced dataset

logreg1 <- glm(y ~., data=train,family='binomial')
pred1 <- predict(logreg1, newdata=test)
pred1[1:5]

library(ROSE)
accuracy.meas(test$y, pred1)
'''
precision: 0.689 means the model has correcctly predicted 70%  of its positive 
predictions
recall: 0.263 means out of acctual positives , mmodel has only predicted 26% as
positives
F: 0.190 this balances precision and recall. But only 19%, so very less
'''
confusion1 <- table(test$y, pred1>=0.5)
table(test$y)
accuracy1 <- sum(diag(confusion1))/sum(confusion1)
accuracy1
# though accuracy is 90%, the results are highly deceiving since minority class 
# holds minimum effect on overall accuracy.

################### Handling imbalanced data
''' ML algorithms believe that the datasset has balanced class distributions.
When we have imbalanced datasets, the results get biased towars majority class.
There are various techniques to deal with such datasets. we will use them and
build our mmodels and evaluate their performances .
We will be using ROSE package. We will use train dataset to build the newdata'''

library(ROSE)
head(train)
# OVERSAMPLING
# we oversample the minority class
table(train$y)
# no=23953, yes=3173, total=27126
23953+3173
23953+23953 # 47906
bank_over <- ovun.sample(y~.,data=train,method='over',N=47906,seed=123)$data
table(bank_over$y)
#    no   yes 
#  23953 23953

# Undersampling: we undersample majority class without replacement
3173+3173 #6346
bank_under <- ovun.sample(y~.,data=train,method='under',N=6346,seed=123)$data
table(bank_under$y)

# both
bank_both <- ovun.sample(y~.,data=train,method='both',N=nrow(train),
                         p=0.5,seed=123)$data
table(bank_both$y)
#    no   yes 
#  13713 13413  total=27126 same as total records in train data

# Synthetic data
bank_rose <- ROSE(y~., data=train, seed=123)$data
table(bank_rose$y)
#    no   yes 
#  13713 13413

#### model building using the new datasets
# computing model using each data and evaluating its accuracy

logreg_over <- glm(y ~., data=bank_over,family='binomial')
logreg_under <- glm(y ~., data=bank_under,family='binomial')
logreg_both <- glm(y ~., data=bank_both,family='binomial')
logreg_rose <- glm(y ~., data=bank_rose,family='binomial')

# making predictions using test data
pred_over <- predict(logreg_over, newdata=test)
pred_under <- predict(logreg_under, newdata=test)
pred_both <- predict(logreg_both, newdata=test)
pred_rose <- predict(logreg_rose, newdata=test)

# ROC and AUC
# plotting all ROC curves
roc.curve(test$y, pred_over) # AUC is 0.910
roc.curve(test$y, pred_under,add=TRUE,col='green') # AUC is 0.908
roc.curve(test$y, pred_both,add=TRUE,col='red') # AUC is 0.908
roc.curve(test$y, pred_rose,add=TRUE,col='blue') # AUC is 0.904
legend(0.6,0.45, c('over','under','both','rose'),lty=c(1,1),
       lwd=c(2,2),col=c('black','green','red','blue'))

# we are getting almost same accuracy from all sampling methods. Still
# highest accuracy is from oversampling technique

bank_holdout <- ROSE.eval(y ~ ., data =train,learner = glm,method.assess="holdout",
                          control.learner=list(family=binomial),seed = 123)
bank_holdout # auc is 0.908
''' We see that our accuracy retains at 0.908 and shows that our predictions
are not suffering from high variance.'''

'''
CONCLUSIONS

The business problem is to find if the customer has subscribed to a term deposit 
or not. Since the output variable is categorical we have used logistic 
regression technique.

We have found that the dataset is imbalanced. So we have used various sampling
methods and We have prepared various models. The accuracy was almost same
with all sampling techniques. But highest accuracy was obtained from data 
prepared using oversampling technique.  

We used holdout method to check if predictions have high variance. We obtained
same accuracy, suggesting that our model is good.

'''



