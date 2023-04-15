#Customer conversion prediction : - GUVI data science project by Prathamesh Kadam

#importing the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics # evaluation metrics

dataset = pd.read_csv(#CSV file_path) #reading the dataset

print(dataset.shape)
dataset.head()

"""#Data Cleaning"""

dataset.describe()

dataset.isnull().sum() #checking for null values

dataset.dtypes #checking for data types

dataset = dataset.drop_duplicates() #removing duplicate rows

dataset.shape

#checking for imbalanced data
target_count = dataset.y.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion of class 0 is ', round(target_count[0] * 100 / (target_count[1] + target_count[0]), 2),'%')

target_count.plot(kind='bar', title='Count (y)');

"""Removing Outliers"""

#age column
iqr = dataset.age.quantile(0.75) - dataset.age.quantile(0.25)
upper_th = dataset.age.quantile(0.75) + (1.5 * iqr) # q3 + 1.5iqr
lower_th = dataset.age.quantile(0.25) - (1.5 * iqr) # q1 - 1.5iqr
dataset['age'] = dataset['age'].clip(lower_th,upper_th)
#dataset.head()

#dur column
iqr = dataset.dur.quantile(0.75) - dataset.dur.quantile(0.25)
upper_th = dataset.dur.quantile(0.75) + (1.5 * iqr) # q3 + 1.5iqr
lower_th = dataset.dur.quantile(0.25) - (1.5 * iqr) # q1 - 1.5iqr
dataset['dur'] = dataset['dur'].clip(lower_th,upper_th)
#dataset.head()

#num_calls column
iqr = dataset.num_calls.quantile(0.75) - dataset.num_calls.quantile(0.25)
upper_th = dataset.num_calls.quantile(0.75) + (1.5 * iqr) # q3 + 1.5iqr
dataset['num_calls'] = dataset['num_calls'].clip(lower_th,upper_th)
#dataset.describe()

#day column
iqr = dataset.day.quantile(0.75) - dataset.day.quantile(0.25)
upper_th = dataset.day.quantile(0.75) + (1.5 * iqr) # q3 + 1.5iqr
lower_th = dataset.day.quantile(0.25) - (1.5 * iqr) # q1 - 1.5iqr
dataset['day'] = dataset['day'].clip(lower_th,upper_th)

dataset.describe()

"""#EDA"""

cat_vr = dataset.select_dtypes(include=["object"]).columns #categorical variables 
for column in cat_vr:
  plt.figure(figsize=(20,5))
  plt.subplot(121)
  dataset[column].value_counts().plot(kind = 'bar')
  plt.xlabel(column)
  plt.ylabel("No. of customers")
  plt.title(column)

#age column
plt.figure(figsize=(30,10))
sns.countplot(data = dataset, x='age', hue="y")

#num_calls column
plt.figure(figsize=(20,10))
sns.countplot(data = dataset, x='num_calls', hue="y")

#Replacing unknown values in education_qual and job columns with mode
dataset = dataset.copy()
dataset['education_qual'] = dataset['education_qual'].replace('unknown', dataset['education_qual'].mode()[0]) #imputing education_qual unknown
dataset['job'] = dataset['job'].replace('unknown', dataset['job'].mode()[0]) #imputing job unknown

#renaming unknown values in call_type and prev_outcome 
dataset['call_type'] = dataset['call_type'].replace('unknown', 'unknown_call_type')
dataset['prev_outcome'] = dataset['prev_outcome'].replace('unknown', 'unknown_prev_outcome')

dataset.columns

"""#Encoding of categorical variables"""

#encoding categorical variables

for i in cat_vr[:-1]:

  one_hot = pd.get_dummies(dataset[i]) #one hot encoding
  dataset = dataset.drop(i,axis = 1)
  dataset = dataset.join(one_hot)

dataset["y"] = dataset["y"].map({"yes":1,"no":0}) #encoding binary class data (run only once)

dataset.columns

"""#Splitting the data"""

dataset

x = dataset[['age', 'day', 'dur', 'num_calls', 'admin.', 'blue-collar',
       'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed',
       'services', 'student', 'technician', 'unemployed', 'divorced',
       'married', 'single', 'primary', 'secondary', 'tertiary', 'cellular',
       'telephone', 'unknown_call_type', 'apr', 'aug', 'dec', 'feb', 'jan',
       'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep', 'failure', 'other',
       'success', 'unknown_prev_outcome']].values #features
y = dataset[['y']].values #target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size = 0.3, random_state = 101)

y_test

"""#Balancing the data"""

#balancing the data
from imblearn.combine import SMOTEENN
smt = SMOTEENN(sampling_strategy='all')
x_train, y_train = smt.fit_resample(x_train, y_train)

"""#Scaling the data"""

#scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

"""#Models"""

#logistic regression
from sklearn.linear_model import LogisticRegression #importing the logistic regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train,y_train) #fitting the model
y_pred = logistic_regression.predict(x_test)
y_pred

#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(6) #using CV score it is observed that k=6 has best value
knn.fit(x_train,y_train)
knn.score(x_test,y_test)
y_pred = knn.predict(x_test)
y_pred

# for i in [1,2,3,4,5,6,7,8,9,10,20,50]:
#   knn = KNeighborsClassifier(i) #initialising the model
#   knn.fit(x_train,y_train) # training the model
#   print("K value  : " , i, " train score : ", knn.score(x_train,y_train) , " cv score : ", np.mean(cross_val_score(knn, x_train, y_train, cv=10, scoring = "roc_auc")))

#Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
y_pred

"""#Evaluating the models"""

#Evaluating Logisticregression
from sklearn.metrics import roc_auc_score
#logistic_regression.score(x_test,y_test)
#f1_score(y_test,y_pred)
y_pred = logistic_regression.predict_proba(x_test)
auroc = roc_auc_score(y_test,y_pred[:,1])
print("Test set auc: {:.2f}".format(auroc))

#Evaluating Decisiontreeclassifier
y_pred = dt.predict(x_test) # Model's predictions
acc = roc_auc_score(y_test, y_pred)
print("Test set auc: {:.2f}".format(acc))

#Evaluating KNeighbours Classifier
y_pred = knn.predict(x_test)
auroc = roc_auc_score(y_test,y_pred)
print("Test set auc: {:.2f}".format(auroc))

"""#Ensembling"""

#VotingClassifier
from sklearn.ensemble import VotingClassifier 
from sklearn import tree
model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)
model3 = KNeighborsClassifier(3)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2),('knn',model3)], voting='soft')
model.fit(x_train,y_train)
pred = model.predict(x_test)
auroc = roc_auc_score(y_test,model.predict_proba(x_test)[:,1])
print("Test set auc: {:.2f}".format(auroc))

#bagging classifier
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(max_depth = 1, random_state=1), n_estimators=100)
model.fit(x_train, y_train)
auroc = roc_auc_score(y_test,model.predict_proba(x_test)[:,1])
print("Test set auc: {:.2f}".format(auroc))

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators= 100, max_depth = 3, max_features='sqrt')
rf.fit(x_train,y_train)
pred = rf.predict(x_test)
auroc = roc_auc_score(y_test,rf.predict_proba(x_test)[:,1])
print("Test set auc: {:.2f}".format(auroc))

#XGBOOST Classifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import numpy as np
for lr in [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.2,0.5,0.7,1]:
  model = xgb.XGBClassifier(learning_rate = lr, n_estimators=100, verbosity = 0) # initialise the model
  model.fit(x_train,y_train) #train the model
  model.score(x_test, y_test) # scoring the model - r2 squared
  print("Learning rate : ", lr, " Train score : ", model.score(x_train,y_train), " Cross-Val score : ", np.mean(cross_val_score(model, x_train, y_train, cv=10)))

model = xgb.XGBClassifier(learning_rate = 0.04, n_estimators=100)
model.fit(x_train,y_train) #train the model
model.score(x_test, y_test) # scoring the model - r2 squared

"""#Important Feature"""

importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
feat_labels = dataset.columns[1:]

for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                    feat_labels[sorted_indices[f]],
                    importances[sorted_indices[f]]))
