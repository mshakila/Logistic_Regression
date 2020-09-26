# Logistic Regression Assignment on affairs dataset

# Business Problem: To classify persons having and not having affairs

# Data collection: 
affairs_raw <- read.csv("D:\\Logistic_Regression_Assignment\\affairs.csv")
# 601 records of 9 variables

# some info about the dataset
'''
The details of variables in this dataset is given below:
Gender Identification (male, female)
Age
Number of Years Married
Children (yes, no)
How Religious, from 1 (not religious) to 5 (very religious)
Education
Occupation, from 1 to 7, according to hollingshead classification
Self Rating of Marriage, from 1 (very unhappy) to 5 (very happy)
Number of Affairs in the Past Year
'''
'''
since affairs is a binary variable (person has affairs or does not have), we can
fit binary Logistic Regression to classify the data
'''

head(affairs_raw)
summary(affairs_raw)
str(affairs_raw)
names(affairs_raw)
#  "affairs"       "gender"        "age"           "yearsmarried"  "children"     
#  "religiousness" "education"     "occupation"    "rating" 

attach(affairs_raw)
table(affairs)
table(gender)
table(children)


# affairs is the outcome variable and a person has affairs or not 
# if he/she has no affairs using 0 and has affairs using 1 to represent the same.
affairs_raw1 <- affairs_raw
affairs_raw1$affairs <- ifelse(affairs_raw1$affairs>0,1,0)
# also class (data type) of affairs is given as int, let us convert it to factor
class(affairs_raw1$affairs)
affairs_raw1$affairs <- as.factor(affairs_raw1$affairs)

table(affairs_raw1$affairs)
#   0   1 
#  451 150

prop.table(table(affairs_raw1$affairs))
#        0        1 
#  0.750416 0.249584 
# there are 75% persons not having affairs and 25% who have affairs

# there are 2 categorical variables. to know how R dummifies them 
contrasts(affairs_raw1$gender)
contrasts(affairs_raw1$children)
# in gender, female is 0, it will be used as reference
# in children, no is 0, it will be used as reference.

# checking for missing values
sum(is.na(affairs_raw1))
library(Amelia)
missmap(affairs_raw1,main='Observed vs missing values')
# the whole plot is blue (for observed values), no missing values

# Building Logistic Regression model
logreg1 <- glm(affairs ~ ., data=affairs_raw1,family='binomial')

summary(logreg1)
# null deviance is 675, Resid dev is 609 and AIC is 627
'''
in logit model, response variable is given as natural log of odds. When rating 
increases by 1 unit, log odds reduces by 0.47. 
'''
# To calculate the odds ratio manually 
exp(coef(logreg1))


# analyze the table of deviance uisng anova
anova(logreg1, test='Chisq')
'''
the difference in null and resid. dev shows how our model is doing against
the null model. greater the difference, better is our model.
from table we can see drop in deviancce when each variable is added one at
a time. 
'''

# gender, children, education and occupation are not significant
# let us remove these and build another model

logreg2 <- glm(affairs ~ age+yearsmarried+education+occupation, data=affairs_raw1,family='binomial')

summary(logreg2)
# null deviance is 675, Resid dev is 657 and AIC is 667
# AIC is more and even resid. deviance is more as compared to previous model.
# so logreg1 is a better model

# let us predict the values, we get the probabilities 
pred_prob <- logreg1$fitted.values
head(logreg1$fitted.values)
# pred_prob=predict(logreg1,type=c("response"),affairs_raw1)

# let us now look at the accuracy measures
# confusion matrix
confusion <- table(pred_prob>=0.5, affairs_raw1$affairs)
confusion
table(affairs_raw1$affairs)

Accuracy<-sum(diag(confusion))/sum(confusion)
Accuracy # 75%
# but we can see that the model has correctly predicted No as NO.
# all Yes are wrongly predicted by the model.
# here alpha error (false positive) is zero, which is very good.

# ROC Curve 
library(ROCR)
rocrpred<-prediction(pred_prob, affairs_raw1$affairs)
rocrperf<-performance(rocrpred,'tpr','fpr')
plot(rocrperf,colorize=T)
# More area under the ROC Curve better is the logistic regression model obtained

# auc
auc <- performance(rocrpred,measure='auc')
auc <- auc@y.values[[1]]
auc # 0.61
# the auc is 61%. area should be cclose to 1 and far from 0.5, to be a good
# model. 

#########################   Build model using train 
library(caTools)
set.seed(100)
split_data <- sample.split(affairs_raw1$affairs, SplitRatio=0.8)
train <- subset(affairs_raw1, split_data==TRUE)
test <- subset(affairs_raw1, split_data==FALSE)

logreg3 <- glm(affairs ~ ., data=train,family='binomial')
summary(logreg3)
# null dev is 540, resid dev is 482 and AIC is 500 which is far less than 
#previous models, hence a better model.

# training accuracy
pred_prob3 <- logreg3$fitted.values
# confusion matrix
confusion3 <- table(pred_prob3>=0.5, train$affairs)
confusion3
table(train$affairs)
train_Accuracy<-sum(diag(confusion3))/sum(confusion3)
train_Accuracy # 76.7%

# testing accuracy
pred_test <- predict(logreg3,test,type='response')
pred_test[1:5]
confusion_test <- table(pred_test>=0.5, test$affairs)
confusion_test
table(test$affairs)
test_Accuracy<-sum(diag(confusion_test))/sum(confusion_test)
test_Accuracy # 75.8%
# both train and test accuracy are almost same. 

# ROCR
rocrpred3<-prediction(pred_prob3, train$affairs)
rocrperf3<-performance(rocrpred3,'tpr','fpr')
plot(rocrperf3,colorize=T)
abline(a=0, b=1, col='black')
# auc
auc3 <- performance(rocrpred3,measure='auc')
auc3 <- auc3@y.values[[1]]
auc3 # 72.4%

# auc for test data
rocrpred3_test <- prediction(pred_test,test$affairs)
auc3_test <- performance(rocrpred3_test, measure='auc')
auc3_test <- auc3_test@y.values[[1]]
auc3_test # 65.85%

# the auc for previous model (without splitting the data) was 61%. After splitting
# the data it has increases for both train data and test data.

#let us compute optimal cutoff value
library(InformationValue)
opt_cutoff = optimalCutoff(test$affairs,pred_test)[1]
opt_cutoff # 0.5388419

confusion_test1 <- table(pred_test>=0.54, test$affairs)
confusion_test1
table(test$affairs)
test_Accuracy1<-sum(diag(confusion_test1))/sum(confusion_test1)
test_Accuracy1 # 77.5%

# for cut_off 0.5, test-accuracy is 75.8%
# for cut_off 0.5388, test-accuracy is 77.5%
# for cut_off 0.54, test-accuracy is 77.5%
# for cut_off 0.6, test-accuracy is 75%

'''
the accuracy is decreasing bothways if we move from the optimal cutoff value.
The false positive has decreased when we consider cuoff value.
'''
'''
CONCLUSIONS

The business problem is to classify persons having affairs from others not having
affairs. Since the output variable is categorical we have used logistic 
regression technique.

We have prepared various models, and chosen the better model based on
accuracy, ROC and auc.  we have also used optimal cut-off value to reduce
misclassification errors.

To increase our accuracy and decrease misclassification, we need more records.

'''



