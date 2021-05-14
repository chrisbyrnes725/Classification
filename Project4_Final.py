#!/usr/bin/env python
# coding: utf-8

# In[192]:


import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[193]:


df_class = pd.read_csv('/Users/chrisbyrnes/Metis/Classification/ClassDataCSV.csv')


# In[194]:


df_class.info()


# In[195]:


df_class.head()


# In[196]:


X = df_class[['SciCont','SciExpert','EnePrior','IncRel','CliImp', 'MedProf', 'ClinTrials', 'USHelp',
              'CovidThreat', 'SocDist', 'Vaccine','Female','PolLean', 'COVED_Bin','PolLean']]
y = df_class['OneYear']


# In[197]:


print(X.shape)
print(y.shape)


# In[198]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)


# In[199]:


X_scaled, X_test, y, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=5)


# In[200]:


print(X_scaled.shape)
print(y.shape)


# In[201]:


knn = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='recall')
print(scores)
print(scores.mean())


# In[202]:


k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='recall')
    k_scores.append(scores.mean())
print(k_scores)


# In[203]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Recall')


# In[204]:


k_range = range(1, 100)
print(k_range)


# In[205]:


param_grid = dict(n_neighbors=k_range)
print(param_grid)


# In[206]:


grid = GridSearchCV(knn, param_grid, cv=10, scoring='recall')


# In[207]:


grid.fit(X_scaled, y);


# In[208]:


print("Best params: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)
print("Best score: ", grid.best_score_)


# In[209]:


y_pred = grid.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.recall_score(y_test, y_pred))
print(metrics.precision_score(y_test, y_pred))


# In[210]:


print("kNN confusion matrix: \n\n", confusion_matrix(y_test, grid.predict(X_test)))


# In[211]:


lr = LogisticRegression()
lr.fit(X_scaled, y)
scores = cross_val_score(lr, X_scaled, y, cv=10, scoring='recall')
print(scores)


# In[212]:


lr = LogisticRegression(class_weight = 'balanced')
lr.fit(X_scaled, y)
scores = cross_val_score(lr, X_scaled, y, cv=10, scoring='recall')
print(scores)


# In[213]:


y_pred = lr.predict(X_test)
print(metrics.recall_score(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))


# In[214]:


print("Logistic Regression confusion matrix: \n\n", confusion_matrix(y_test, lr.predict(X_test)))


# In[215]:


y_predict = (lr.predict_proba(X_test)[:,1] > 0.3)
print("Threshold of 0.4:")
print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_test, y_predict), 
                                                     recall_score(y_test, y_predict)))


# In[216]:


print("Logistic Regression confusion matrix: \n\n", confusion_matrix(y_test, lr.predict(X_test)))


# In[217]:


sns.heatmap(df_class.corr(), cmap="coolwarm",vmin=-1, vmax=1)
plt.show()


# In[218]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[219]:


decisiontree = DecisionTreeClassifier(max_depth=4)
decisiontree.fit(X_scaled, y)
decisiontree.score(X_test, y_test)
scores = cross_val_score(decisiontree, X_scaled, y, cv=5, scoring='recall')
print(scores)


# In[220]:


randomforest = RandomForestClassifier(n_estimators=100)
randomforest.fit(X_scaled, y)
scores = cross_val_score(randomforest, X_scaled, y, cv=5, scoring='recall')
print(scores)


# In[221]:


import imblearn.over_sampling


# In[222]:


X = df_class[['SciCont','SciExpert','EnePrior','IncRel','CliImp', 'MedProf', 'ClinTrials', 'USHelp',
              'CovidThreat', 'SocDist', 'Vaccine','Female','PolLean']]
y = df_class['OneYear']


# In[223]:


X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
n_pos = np.sum(y_tr == 1)
n_neg = np.sum(y_tr == 0)
ratio = {1 : n_pos * 6, 0 : n_neg} 
ROS = imblearn.over_sampling.RandomOverSampler(sampling_strategy = ratio, random_state=5)
X_tr_rs, y_tr_rs = ROS.fit_resample(X_tr,y_tr)


# In[224]:


lr = LogisticRegression(solver='liblinear') 
lr.fit(X_tr, y_tr)

lr_os = LogisticRegression(solver='liblinear') 
lr_os.fit(X_tr_rs, y_tr_rs)


# In[225]:


y_pred = lr.predict(X_test)
print(metrics.recall_score(y_test, y_pred))

y_pred_os = lr_os.predict(X_test)
print(metrics.recall_score(y_test, y_pred_os))


# In[226]:


print("Logistic Regression confusion matrix: \n\n", confusion_matrix(y_test, lr_os.predict(X_test)))


# In[227]:


from sklearn.naive_bayes import BernoulliNB


# In[228]:


nb = BernoulliNB()
nb.fit(X_tr_rs,y_tr_rs)
nb.score(X_test,y_test)


# In[229]:


y_pred = nb.predict(X_test)
print(metrics.recall_score(y_test, y_pred))
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.precision_score(y_test,y_pred))
print(metrics.f1_score(y_test,y_pred))


# In[230]:


print("Naive Bayes confusion matrix: \n\n", confusion_matrix(y_test, nb.predict(X_test)))


# In[231]:


###### Work on modifying chosen logistic regression model


# In[347]:


X = df_class[['SciExpert','CovidThreat', 'Vaccine', 'COVED_Bin','PolLean']]
y = df_class['OneYear']


# In[348]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)


# In[349]:


X_scaled, X_test, y, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=5)


# In[350]:


lr = LogisticRegression(class_weight = 'balanced',C = 1000)
lr.fit(X_scaled, y)
scores = cross_val_score(lr, X_scaled, y, cv=10, scoring='recall')
print(scores)


# In[351]:


y_pred = lr.predict(X_test)
print(metrics.recall_score(y_test, y_pred))

print(lr.coef_)


# In[352]:


from sklearn.metrics import roc_auc_score, roc_curve

fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])


# In[353]:


plt.plot(fpr, tpr,lw=2)
plt.plot([0,1],[0,1],c='violet',ls='--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])


plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve');
print("ROC AUC score = ", roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]))


# In[354]:


print(metrics.precision_score(y_test, y_pred))
print(metrics.f1_score(y_test, y_pred))


# In[355]:


smote = imblearn.over_sampling.SMOTE(sampling_strategy=ratio, random_state = 42)
    
X_scaled_smote, y_smote = smote.fit_resample(X_scaled, y)

lr_smote = LogisticRegression(solver='liblinear') 
lr_smote.fit(X_scaled_smote, y_smote)

print(metrics.f1_score(y_test, lr_smote.predict(X_test)))
print(metrics.recall_score(y_test, lr_smote.predict(X_test)))      


# In[356]:


y_predict = (lr.predict_proba(X_test)[:,1] > 0.45)
print("Threshold of 0.45:")
print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_test, y_predict), 
                                                     recall_score(y_test, y_predict)))


# In[370]:


from sklearn.metrics import precision_score, recall_score, accuracy_score

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

X_val, y_val = X_test, y_test 

thresh_ps = np.linspace(.10,.50,1000)
model_val_probs = lr.predict_proba(X_val)[:,1] # positive class probs
f1_scores, prec_scores, rec_scores, acc_scores = [], [], [], []
for p in thresh_ps:
    model_val_labels = model_val_probs >= p
    f1_scores.append(f1_score(y_val, model_val_labels))    
    prec_scores.append(precision_score(y_val, model_val_labels))
    rec_scores.append(recall_score(y_val, model_val_labels))
    acc_scores.append(accuracy_score(y_val, model_val_labels))
    
plt.plot(thresh_ps, f1_scores)
plt.plot(thresh_ps, prec_scores)
plt.plot(thresh_ps, rec_scores)
plt.plot(thresh_ps, acc_scores)

plt.title('Metric Scores vs. Positive Class Decision Probability Threshold')
plt.legend(['F1','Precision','Recall','Accuracy'],  loc='best')
plt.xlabel('P threshold')

#bbox_to_anchor=(1.05, 0),
best_f1_score = np.max(f1_scores) 
best_thresh_p = thresh_ps[np.argmax(f1_scores)]

print('Logistic Regression Model best F1 score %.3f at prob decision threshold >= %.3f' 
      % (best_f1_score, best_thresh_p))
plt.savefig('Logistic_Metrics.PNG', transparent=True)


# In[372]:


y_predict = (lr.predict_proba(X_test)[:,1] > 0.45)
print("Threshold of 0.45:")
print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_test, y_predict), 
                                                     recall_score(y_test, y_predict)))


# In[359]:


print("Logistic Regression confusion matrix: \n\n", confusion_matrix(y_test, lr.predict(X_test)))

