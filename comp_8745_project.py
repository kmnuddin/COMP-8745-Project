
# coding: utf-8

# # COMP 8745 Final Project

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold


# In[6]:


data_train = pd.read_csv('training.txt', sep='\t', header=None,engine='python', error_bad_lines=False, warn_bad_lines=False)
data_train.columns = ['Reviews','Ratings']

data_test = pd.read_csv('test.txt', sep='\t',header=None,engine='python', error_bad_lines=False, warn_bad_lines=False)
data_test.columns = ['Reviews', 'Ratings']

X_train = data_train['Reviews']
y_train = data_train['Ratings']

X_test = data_test['Reviews']
y_test = data_test['Ratings']


# In[7]:


positive_text = pd.read_csv('positive-words.txt', sep='delimiter', header=None,engine='python', 
                            error_bad_lines=False, warn_bad_lines=False)

positive_text.columns = ['sentiment_words']
pos_data = positive_text['sentiment_words']

negative_text = pd.read_csv('negative-words.txt', sep='delimiter', header=None,engine='python', 
                            error_bad_lines=False, warn_bad_lines=False)

negative_text.columns = ['sentiment_words']
neg_data = negative_text['sentiment_words']

filter_data = pos_data.append(neg_data)


# In[55]:


kf = KFold(n_splits=5)


# ## Implementation of bag of words model using TfidfVectorizer

# In[8]:


vectorizer = TfidfVectorizer(stop_words='english')
vector_train = vectorizer.fit_transform(X_train)

vector_test = vectorizer.transform(X_test)


# # Task 1

# ## Neural Nets

# In[3]:


hidden_layers = [(25, 25), (25, 25, 25), (30, 30, 30), (40, 40, 40), (40, 40, 40, 40), (50, 50, 50, 50), (100, 100, 100, 100)] 


# ### 5-Fold Cross Validation

# In[122]:


fold = 1;
for train_index, test_index in kf.split(X_train):
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    for layer in hidden_layers:
        nn = MLPClassifier(hidden_layer_sizes=layer, learning_rate_init=0.01)
        nn.fit(X_traincv, y_traincv)
        y_predcv = nn.predict(X_testcv)
        #accuracy_scores.append(metrics.accuracy_score(y_testcv, y_predcv))
        print('--------- Scores for hidden layers {}---------'.format(layer))
        print(metrics.classification_report(y_testcv, y_predcv))
    fold = fold + 1


# In[39]:


for layer in hidden_layers:
    nn = MLPClassifier(hidden_layer_sizes=layer, learning_rate_init=0.01)
    avg_precision = cross_val_score(nn, vector_train, y_train, cv=5, scoring='precision_micro')
    avg_recall = cross_val_score(nn, vector_train, y_train, cv=5, scoring='recall_micro')
    avg_f1 = cross_val_score(nn, vector_train, y_train, cv=5, scoring='f1_micro') 
    print('-------- Average Scores for Layer {}'.format(layer))
    print_cv_score()


# In[38]:


def print_cv_score():
    print("Avg Precision: % .2f" % avg_precision.mean())
    print("Avg Recall: % .2f" % avg_recall.mean())
    print("Avg f1 score: % .2f" % avg_f1.mean())    


# ## Naive Bayes

# ### 5-Fold Cross Validation

# In[110]:


fold = 1
for train_index, test_index in kf.split(X_train):z
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    nb = MultinomialNB()
    nb.fit(X_traincv, y_traincv)
    y_predcv = nb.predict(X_testcv)
    fold = fold + 1
    print(metrics.classification_report(y_testcv, y_predcv))


# In[40]:


nb = MultinomialNB()
avg_precision = cross_val_score(nb, vector_train, y_train, cv=5, scoring='precision_micro')
avg_recall = cross_val_score(nb, vector_train, y_train, cv=5, scoring='recall_micro')
avg_f1 = cross_val_score(nb, vector_train, y_train, cv=5, scoring='f1_micro') 


# In[41]:


print_cv_score()


# ## Logistic Regression

# ### 5-Fold Cross Validation

# #### Logistic Regression using L1 Regularization

# In[114]:


C = [0.01, 0.1, 1, 10, 100]
fold = 1;
print('--------------- Logistic Regression using L1 Regularization ----------------')
for train_index, test_index in kf.split(X_train):
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    for c in C:
        lgr_l1 = LogisticRegression(penalty='l1', C=c)
        lgr_l1.fit(X_traincv, y_traincv)
        y_predcv = lgr_l1.predict(X_testcv) 
        print('---------- Scores using strength parameter C = {} -----------'.format(c))
        print(metrics.classification_report(y_testcv, y_predcv))
    fold = fold + 1


# In[42]:


C = [0.01, 0.1, 1, 10, 100]
for c in C:
    lgr_l1 = LogisticRegression(penalty='l1', C=c)
    avg_precision = cross_val_score(lgr_l1, vector_train, y_train, cv=5, scoring='precision_micro')
    avg_recall = cross_val_score(lgr_l1, vector_train, y_train, cv=5, scoring='recall_micro')
    avg_f1 = cross_val_score(lgr_l1, vector_train, y_train, cv=5, scoring='f1_micro') 
    print('----------- Average Score {} -----------'.format(c))
    print_cv_score()


# #### Logistic Regression using L2 Regularization

# In[115]:


fold = 1;
print('--------------- Logistic Regression using L2 Regularization ----------------')
for train_index, test_index in kf.split(X_train):
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    for c in C:
        lgr_l1 = LogisticRegression(penalty='l2', C=c)
        lgr_l1.fit(X_traincv, y_traincv)
        y_predcv = lgr_l1.predict(X_testcv) 
        print('---------- Scores using strength parameter C = {} -----------'.format(c))
        print(metrics.classification_report(y_testcv, y_predcv))
    fold = fold + 1


# In[43]:


for c in C:
    lgr_l1 = LogisticRegression(penalty='l2', C=c)
    avg_precision = cross_val_score(lgr_l1, vector_train, y_train, cv=5, scoring='precision_micro')
    avg_recall = cross_val_score(lgr_l1, vector_train, y_train, cv=5, scoring='recall_micro')
    avg_f1 = cross_val_score(lgr_l1, vector_train, y_train, cv=5, scoring='f1_micro')
    print('----------- Average Score {} -----------'.format(c))
    print_cv_score()


# ## AdaBoosting

# ### 5-Fold Cross Validation

# In[103]:


fold = 1
for train_index, test_index in kf.split(X_train):
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    for n in range(40, 110, 10):
        adb = AdaBoostClassifier(n_estimators=n)
        adb.fit(X_traincv, y_traincv)
        y_predcv = adb.predict(X_testcv)
        print('--------- Scores for {} estimators---------'.format(n))
        print(metrics.classification_report(y_testcv, y_predcv))
    fold = fold + 1


# In[44]:


for n in range(40, 110, 10):
    adb = AdaBoostClassifier(n_estimators=n)
    avg_precision = cross_val_score(adb, vector_train, y_train, cv=5, scoring='precision_micro')
    avg_recall = cross_val_score(adb, vector_train, y_train, cv=5, scoring='recall_micro')
    avg_f1 = cross_val_score(adb, vector_train, y_train, cv=5, scoring='f1_micro')
    print('----------- Average Score {} -----------'.format(n))
    print_cv_score()


# ## SVM

# ### 5-Fold Cross Validation

# #### SVC using Gausian Kernel

# In[116]:


fold = 1
for train_index, test_index in kf.split(X_train):
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    for c in C:
        svc = SVC(C=c, kernel='rbf')
        svc.fit(X_traincv, y_traincv)
        y_predcv = svc.predict(X_testcv)
        print('---------- Scores using C = {} -----------'.format(c))
        print(metrics.classification_report(y_testcv, y_predcv))
    fold = fold + 1


# In[45]:


for c in C:
    svc = SVC(C=c, kernel='rbf')
    avg_precision = cross_val_score(svc, vector_train, y_train, cv=5, scoring='precision_micro')
    avg_recall = cross_val_score(svc, vector_train, y_train, cv=5, scoring='recall_micro')
    avg_f1 = cross_val_score(svc, vector_train, y_train, cv=5, scoring='f1_micro')
    print('----------- Average Score for C: {} -----------'.format(c))
    print_cv_score()


# #### SVC using Poly Kernel

# In[258]:


fold = 1
degree = [1, 2, 3, 5, 10]
for train_index, test_index in kf.split(X_train):
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    for d in degree:
        svc = SVC(C=1000, kernel='poly', degree=d)
        svc.fit(X_traincv, y_traincv)
        y_predcv = svc.predict(X_testcv)
        print('---------- Scores using poly kernel of degree = {} -----------'.format(d))
        print(metrics.classification_report(y_testcv, y_predcv))
    fold = fold + 1


# In[47]:


degree = [1, 2, 3, 5, 10]
for d in degree:
    svc = SVC(C=1000, kernel='poly', degree=d)
    avg_precision = cross_val_score(svc, vector_train, y_train, cv=5, scoring='precision_micro')
    avg_recall = cross_val_score(svc, vector_train, y_train, cv=5, scoring='recall_micro')
    avg_f1 = cross_val_score(svc, vector_train, y_train, cv=5, scoring='f1_micro')
    print('----------- Average Score for degree: {} -----------'.format(d))
    print_cv_score()


# # Task 2

# ## Imlementation of TfidfVectorizer using sentiment words

# In[241]:


vectorizer_filter = TfidfVectorizer(stop_words='english')
vectorizer_filter.fit_transform(filter_data)

vectorizer = TfidfVectorizer(stop_words='english', vocabulary=vectorizer_filter.get_feature_names())
vector_train = vectorizer.fit_transform(X_train)

vector_test = vectorizer.transform(X_test)


# ## Neural Nets

# ### 5-Fold Cross Validation using sentiment words

# In[174]:


fold = 1;
for train_index, test_index in kf.split(X_train):
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    for layer in hidden_layers:
        nn = MLPClassifier(hidden_layer_sizes=layer, learning_rate_init=0.01)
        nn.fit(X_traincv, y_traincv)
        y_predcv = nn.predict(X_testcv)
        print('-------------Accuracy = {} ------------------'.format(metrics.accuracy_score(y_testcv, y_predcv)))
        print('--------- Scores for hidden layers {}---------'.format(layer))
        print(metrics.classification_report(y_testcv, y_predcv))
    fold = fold + 1


# In[48]:


nn = MLPClassifier(hidden_layer_sizes=(100,100,100,100), learning_rate_init=0.01)
avg_precision = cross_val_score(nn, vector_train, y_train, cv=5, scoring='precision_micro')
avg_recall = cross_val_score(nn, vector_train, y_train, cv=5, scoring='recall_micro')
avg_f1 = cross_val_score(nn, vector_train, y_train, cv=5, scoring='f1_micro')
print_cv_score()


# ## Naive Bayes

# In[232]:


fold = 1
for train_index, test_index in kf.split(X_train):
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    nb = MultinomialNB()
    nb.fit(X_traincv, y_traincv)
    y_predcv = nb.predict(X_testcv)
    print('-------------Accuracy = {} ------------------'.format(metrics.accuracy_score(y_testcv, y_predcv)))
    print(metrics.classification_report(y_testcv, y_predcv))
    fold = fold + 1


# In[49]:


nb = MultinomialNB()
avg_precision = cross_val_score(nb, vector_train, y_train, cv=5, scoring='precision_micro')
avg_recall = cross_val_score(nb, vector_train, y_train, cv=5, scoring='recall_micro')
avg_f1 = cross_val_score(nb, vector_train, y_train, cv=5, scoring='f1_micro')
print_cv_score()


# ## Logistic Regression

# #### Logistic Regression using L1 Regularization

# In[186]:


fold = 1;
print('--------------- Logistic Regression using L1 Regularization ----------------')
for train_index, test_index in kf.split(X_train):
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    for c in C:
        lgr_l1 = LogisticRegression(penalty='l1', C=c)
        lgr_l1.fit(X_traincv, y_traincv)
        y_predcv = lgr_l1.predict(X_testcv) 
        print('-------------Accuracy = {} ------------------'.format(metrics.accuracy_score(y_testcv, y_predcv)))
        print('---------- Scores using strength parameter C = {} -----------'.format(c))
        print(metrics.classification_report(y_testcv, y_predcv))
    fold = fold + 1


# In[50]:


lg_l1 = LogisticRegression(penalty='l1', C=1)
avg_precision = cross_val_score(lg_l1, vector_train, y_train, cv=5, scoring='precision_micro')
avg_recall = cross_val_score(lg_l1, vector_train, y_train, cv=5, scoring='recall_micro')
avg_f1 = cross_val_score(lg_l1, vector_train, y_train, cv=5, scoring='f1_micro')
print_cv_score()


# #### Logistic Regression using L2 Regularization

# In[187]:


fold = 1;
print('--------------- Logistic Regression using L2 Regularization ----------------')
for train_index, test_index in kf.split(X_train):
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    for c in C:
        lgr_l1 = LogisticRegression(penalty='l2', C=c)
        lgr_l1.fit(X_traincv, y_traincv)
        y_predcv = lgr_l1.predict(X_testcv) 
        print('-------------Accuracy = {} ------------------'.format(metrics.accuracy_score(y_testcv, y_predcv)))
        print('---------- Scores using strength parameter C = {} -----------'.format(c))
        print(metrics.classification_report(y_testcv, y_predcv))
    fold = fold + 1


# In[52]:


lg_l1 = LogisticRegression(penalty='l2', C=1)
avg_precision = cross_val_score(lg_l1, vector_train, y_train, cv=5, scoring='precision_micro')
avg_recall = cross_val_score(lg_l1, vector_train, y_train, cv=5, scoring='recall_micro')
avg_f1 = cross_val_score(lg_l1, vector_train, y_train, cv=5, scoring='f1_micro')
print_cv_score()


# ## AdaBoosting

# In[188]:


fold = 1
for train_index, test_index in kf.split(X_train):
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    for n in range(40, 110, 10):
        adb = AdaBoostClassifier(n_estimators=n)
        adb.fit(X_traincv, y_traincv)
        y_predcv = adb.predict(X_testcv)
        print('-------------Accuracy = {} ------------------'.format(metrics.accuracy_score(y_testcv, y_predcv)))
        print('--------- Scores for {} estimators---------'.format(n))
        print(metrics.classification_report(y_testcv, y_predcv))
    fold = fold + 1


# In[53]:


adb = AdaBoostClassifier(n_estimators=100)
avg_precision = cross_val_score(adb, vector_train, y_train, cv=5, scoring='precision_micro')
avg_recall = cross_val_score(adb, vector_train, y_train, cv=5, scoring='recall_micro')
avg_f1 = cross_val_score(adb, vector_train, y_train, cv=5, scoring='f1_micro')
print_cv_score()


# ## SVM

# #### SVC using Gausian Kernel

# In[189]:


fold = 1
for train_index, test_index in kf.split(X_train):
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    for c in C:
        svc = SVC(C=c, kernel='rbf')
        svc.fit(X_traincv, y_traincv)
        y_predcv = svc.predict(X_testcv)
        print('-------------Accuracy = {} ------------------'.format(metrics.accuracy_score(y_testcv, y_predcv)))
        print('---------- Scores using C = {} -----------'.format(c))
        print(metrics.classification_report(y_testcv, y_predcv))
    fold = fold + 1


# #### SVC using Poly Kernel

# In[199]:


fold = 1
for train_index, test_index in kf.split(X_train):
    print('------------ Fold {} -----------'.format(fold))
    X_traincv, X_testcv = vector_train[train_index], vector_train[test_index]
    y_traincv, y_testcv = y_train[train_index], y_train[test_index]
    for d in degree:
        svc = SVC(C=500, kernel='poly', degree=d)
        svc.fit(X_traincv, y_traincv)
        y_predcv = svc.predict(X_testcv)
        print('-------------Accuracy = {} ------------------'.format(metrics.accuracy_score(y_testcv, y_predcv)))
        print('---------- Scores using poly kernel of degree = {} -----------'.format(d))
        print(metrics.classification_report(y_testcv, y_predcv))
    fold = fold + 1


# In[54]:


svc = SVC(C=500, kernel='poly', degree=1)
avg_precision = cross_val_score(svc, vector_train, y_train, cv=5, scoring='precision_micro')
avg_recall = cross_val_score(svc, vector_train, y_train, cv=5, scoring='recall_micro')
avg_f1 = cross_val_score(svc, vector_train, y_train, cv=5, scoring='f1_micro')
print_cv_score()


# # Task 3

# ## Experiment using optimal parameters on test data without filtering

# ## Neural Net

# In[203]:


nn = MLPClassifier(hidden_layer_sizes=(75, 75, 75, 75), learning_rate_init=0.01)
nn.fit(vector_train, y_train)


# In[211]:


y_pred = nn.predict(vector_test)
print('Accuracy : {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('---------- Scores for test data ------------')
print(metrics.classification_report(y_test, y_pred))


# ## Naive Bayes

# In[213]:


nb = MultinomialNB()
nb.fit(vector_train, y_train)


# In[215]:


y_pred = nb.predict(vector_test)
print('Accuracy : {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('---------- Scores for test data ------------')
print(metrics.classification_report(y_test, y_pred))


# ## Logistic Regression

# In[217]:


lgr = LogisticRegression(C=1, penalty='l2')
lgr.fit(vector_train, y_train)


# In[219]:


y_pred = lgr.predict(vector_test)
print('Accuracy : {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('---------- Scores for test data ------------')
print(metrics.classification_report(y_test, y_pred))


# ## AdaBoosting

# In[222]:


adb = AdaBoostClassifier(n_estimators=90)
adb.fit(vector_train, y_train)


# In[224]:


y_pred = adb.predict(vector_test)
print('Accuracy : {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('---------- Scores for test data ------------')
print(metrics.classification_report(y_test, y_pred))


# ## SVC

# In[56]:


svc = SVC(C = 500, kernel='poly', degree=1)
svc.fit(vector_train, y_train)


# In[57]:


y_pred = svc.predict(vector_test)
print('Accuracy : {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('---------- Scores for test data ------------')
print(metrics.classification_report(y_test, y_pred))


# ## Experiment on test data using optimal parameters using sentiment filtering

# ## Neural Net

# In[243]:


nn = MLPClassifier(hidden_layer_sizes=(75, 75, 75, 75), learning_rate_init=0.01)
nn.fit(vector_train, y_train)


# In[245]:


y_pred = nn.predict(vector_test)
print('Accuracy : {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('---------- Scores for test data ------------')
print(metrics.classification_report(y_test, y_pred))


# ## Naive Bayes

# In[246]:


nb = MultinomialNB()
nb.fit(vector_train, y_train)


# In[247]:


y_pred = nb.predict(vector_test)
print('Accuracy : {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('---------- Scores for test data ------------')
print(metrics.classification_report(y_test, y_pred))


# ## Logistic Regression

# In[248]:


lgr = lgr = LogisticRegression(C=1, penalty='l2')
lgr.fit(vector_train, y_train)


# In[252]:


y_pred = lgr.predict(vector_test)
print('Accuracy : {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('---------- Scores for test data ------------')
print(metrics.classification_report(y_test, y_pred))


# ## AdaBoosting

# In[253]:


adb = AdaBoostClassifier(n_estimators=90)
adb.fit(vector_train, y_train)


# In[254]:


y_pred = adb.predict(vector_test)
print('Accuracy : {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('---------- Scores for test data ------------')
print(metrics.classification_report(y_test, y_pred))


# ## SVC

# In[255]:


svc = SVC(C = 100000, kernel='poly', degree=1000)
svc.fit(vector_train, y_train)


# In[257]:


y_pred = svc.predict(vector_test)
print('Accuracy : {}'.format(metrics.accuracy_score(y_test, y_pred)))
print('---------- Scores for test data ------------')
print(metrics.classification_report(y_test, y_pred))

