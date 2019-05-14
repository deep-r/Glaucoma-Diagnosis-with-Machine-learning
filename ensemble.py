# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:13:11 2019

@author: deep
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('extracted_features.csv')
#dataset = dataset.sample(frac=1).reset_index(drop=True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4620].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from time import time
t0 = time()

# Fitting LogReg to the Training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=0)
logreg.fit(X_train, y_train)

# Fitting RF to the Training set
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 400, criterion = "gini")
rf.fit(X_train, y_train)

# Fitting knn to the Training set
from sklearn.neighbors import KNeighborsClassifier  
knn = KNeighborsClassifier(n_neighbors=3, p=3)  
knn.fit(X_train, y_train)

# Fitting svm to the Training set
from sklearn.svm import SVC 
svm = SVC(kernel='rbf')  
svm.fit(X_train, y_train) 


y_pred_logreg = logreg.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_knn = knn.predict(X_test)
ensemble=[]


for i in range(0,181):
    vote = y_pred_logreg[i] + y_pred_rf[i] + y_pred_svm[i] + y_pred_knn[i]
    if vote>=2:
        ensemble.append(1)
    else: ensemble.append(0)

t1 = time()
print ('Run time =',(t1-t0))

y_pred = y_pred_logreg
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
tn, fp, fn, tp = cm.ravel()
sen=tp/(tp+fn)
print '\nSensitivity =', sen
spec= tn/(tn+fp)
print '\nSpecificity =', spec
accuracy = (tp+tn)/(tp+tn+fp+fn)
print '\nAccuracy =', accuracy,'\n'
print "Model = ensemble"
from sklearn.metrics import classification_report, confusion_matrix
print('\nClassification Report')
target_names = ['Glaucomatous', 'Normal']
print(classification_report(y_test, ensemble, target_names=target_names))



########## plot confusion matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[101,13], 
        [17,50]]
df_cm = pd.DataFrame(array, index = ["Normal","Glaucomatous"], columns = ["Normal","Glaucomatous"])

plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt='.3g').set_title('Confusion Matrix for Ensemble Learning',fontsize=15)





from sklearn.metrics import roc_curve, auc
false_positive_rate_en, true_positive_rate_en, thresholds_en = roc_curve(y_test, ensemble)
roc_auc_en = auc(false_positive_rate_en, true_positive_rate_en)

false_positive_rate_knn, true_positive_rate_knn, thresholds_knn = roc_curve(y_test, y_pred_knn)
roc_auc_knn = auc(false_positive_rate_knn, true_positive_rate_knn)

false_positive_rate_svm, true_positive_rate_svm, thresholds_svm = roc_curve(y_test, y_pred_svm)
roc_auc_svm = auc(false_positive_rate_svm, true_positive_rate_svm)

false_positive_rate_rf, true_positive_rate_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
roc_auc_rf = auc(false_positive_rate_rf, true_positive_rate_rf)

false_positive_rate_lr, true_positive_rate_lr, thresholds_lr = roc_curve(y_test, y_pred_logreg)
roc_auc_lr = auc(false_positive_rate_lr, true_positive_rate_lr)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic for all models', fontsize=18)
plt.plot(false_positive_rate_en,true_positive_rate_en,   color='red',   label = 'Ensemble model___________AUC = %0.2f' % roc_auc_en)
plt.plot(false_positive_rate_knn,true_positive_rate_knn, color='green', label = 'Knn model_________________AUC = %0.2f' % roc_auc_knn)
plt.plot(false_positive_rate_svm,true_positive_rate_svm, color='blue',  label = 'SVM model_________________AUC = %0.2f' % roc_auc_svm)
plt.plot(false_positive_rate_rf,true_positive_rate_rf,   color='orange',label = 'Random forest model_______AUC = %0.2f' % roc_auc_rf)
plt.plot(false_positive_rate_lr,true_positive_rate_lr,   color='pink',  label = 'Logistic regression model___AUC = %0.2f' % roc_auc_lr)



plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate', fontsize=18)
plt.xlabel('False Positive Rate', fontsize=18)



