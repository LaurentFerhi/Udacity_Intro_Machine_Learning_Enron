#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from sklearn.cross_validation import train_test_split # cross_validation module depreciated in sklearn v 0.20.4
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# ---------- Main script ------------------------------------------------+

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list =  ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 
'exercised_stock_options', 'expenses', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 
'long_term_incentive', 'other', 'restricted_stock', 'restricted_stock_deferred', 'salary', 
'shared_receipt_with_poi', 'to_messages', 'total_payments', 'total_stock_value', 'ratio_mail_from_poi', 'ratio_mail_to_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Creating a dataframe from the data dictionnary
df = pd.DataFrame(data_dict).T  # dataframe needs to be transpose to have names as index

# Converting all columns to float (all usefull columns can be converted to floats) and replace NaN by 0
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html
df = df.apply(pd.to_numeric, errors = 'coerce').fillna(0)

### Task 2: Remove outliers

# Removing the outliers seen with data_exploration.py script
df.drop('TOTAL', inplace = True)
df.drop('THE TRAVEL AGENCY IN THE PARK', inplace = True)
df.drop('LOCKHART EUGENE E', inplace = True)
df.drop('LAVORATO JOHN J', inplace = True)
df.drop('FREVERT MARK A', inplace = True)
df.drop('KAMINSKI WINCENTY J', inplace = True)
df.drop('MARTIN AMANDA K', inplace = True)
df.drop('WHITE JR THOMAS E', inplace = True)
df.drop('BHATNAGAR SANJAY', inplace = True)

### Task 3: Create new feature(s)

# ratio_mail_from_poi (if from_messages is null, then it returns 0)
df['ratio_mail_from_poi'] = np.where(df['from_messages'] == 0, 0, df['from_poi_to_this_person'] / df['from_messages'])
# ratio_mail_to_poi (if from_messages is null, then it returns 0)
df['ratio_mail_to_poi'] = np.where(df['to_messages'] == 0, 0, df['from_this_person_to_poi'] / df['to_messages'])

# write the dafarame to a dictionnary
cleaned_dict = df.to_dict('index')

### Store to my_dataset for easy export below.
my_dataset = cleaned_dict 

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)

# cross-validator : Stratified ShuffleSplit 
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits = 100, test_size = 0.3, random_state = 42)

# Functions that will be used in the pipeline
scaler = StandardScaler() # Transform features by scaling each feature to a given range (Z-score here)
skb = SelectKBest(f_classif) # Select features according to the k highest scores
pca = PCA() # Principal component analysis


######################
### 1. Gaussian NB ###
######################

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# definition of the pipeline
pipeline = Pipeline(steps = [("SKB",skb),("NaiveBayes",clf)])
param_grid = {"SKB__k":[5,9,10,11,12,15,"all"]} 
grid = GridSearchCV(pipeline, param_grid, verbose = 0, cv = sss)

# training classifier
t0 = time()
grid.fit(features, labels)
print "training time: ", round(time()-t0, 3), "s"

# best classifier using the cross-validator and the Stratified Shuffle Split 
clf = grid.best_estimator_

# predicition with the classifier
t0 = time()
pred = clf.predict(features_test)
print "testing time: ", round(time()-t0, 3), "s"


"""
#Other tested classifiers
########################
### 2. Decision Tree ###
########################

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
# definition of the pipeline
pipeline = Pipeline(steps = [("SKB",skb), ("scaling",scaler), ("PCA",pca), ("dtree",clf)]) 
param_grid = {"SKB__k":[3,4,5,6,7,10,15,"all"], 
              "PCA__n_components":[2,3],
              "PCA__whiten":[True], 
              "dtree__criterion": ["gini", "entropy"], 
              "dtree__min_samples_split": [2, 4, 6, 8, 10],
              "dtree__random_state": [0,40]}
grid = GridSearchCV(pipeline, param_grid, verbose = 0, cv = sss, scoring = 'recall') # choosing recall over precision
# training classifier
t0 = time()
grid.fit(features, labels)
print "training time: ", round(time()-t0, 3), "s"
# best classifier using the cross-validator and the Stratified Shuffle Split 
clf = grid.best_estimator_
# predicition with the classifier
t0 = time()
pred = clf.predict(features_test)
print "testing time: ", round(time()-t0, 3), "s"


###################
### 3. AdaBoost ###
###################

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
# definition of the pipeline
pipeline = Pipeline(steps = [("SKB",skb),("scaling",scaler),("AdaBoost",clf)]) 
param_grid = {"SKB__k":[3,5,10,15,"all"],
              "AdaBoost__n_estimators":(50,100), 
              "AdaBoost__random_state":(0,40)}
grid = GridSearchCV(pipeline, param_grid, verbose = 0, cv = sss)
# training classifier
t0 = time()
grid.fit(features, labels)
print "training time: ", round(time()-t0, 3), "s"
# best classifier using the cross-validator and the Stratified Shuffle Split 
clf = grid.best_estimator_
# predicition with the classifier
t0 = time()
pred = clf.predict(features_test)
print "testing time: ", round(time()-t0, 3), "s"


#################
### 4. Kmeans ###
#################

from sklearn.cluster import KMeans
clf = KMeans()
# definition of the pipeline
pipeline = Pipeline(steps = [("SKB",skb),("scaling",scaler), ("PCA",pca), ("kmean",clf)]) 
param_grid = {"SKB__k":[3,4,5,10,15,"all"], 
              "PCA__n_components":[2,3],
              "PCA__whiten":[True],
              "kmean__n_clusters":(2,), 
              "kmean__random_state":(0,10,40)}
grid = GridSearchCV(pipeline, param_grid, verbose = 0, cv = sss)  
# training classifier
t0 = time()
grid.fit(features, labels)
print "training time: ", round(time()-t0, 3), "s"
# best classifier using the cross-validator and the Stratified Shuffle Split 
clf = grid.best_estimator_
# predicition with the classifier
t0 = time()
pred = clf.predict(features_test)
print "testing time: ", round(time()-t0, 3), "s"
"""


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# List of K best features retained
Kbest_selected = grid.best_estimator_.named_steps['SKB'].get_support()
print "\n > number of retained features:", len([f for f in Kbest_selected if f == True]) # length of the list with only True

# List of importance for each feature
Kbest_importance =  grid.best_estimator_.named_steps['SKB'].scores_


# Dictionnary with the used feature and associated importance
dict_score = {}
for i in range(len(features_list[1:])):
    if Kbest_selected[i] == True:
        dict_score[features_list[1:][i]] = Kbest_importance[i]

print "\n > K best features used with associated importance"
print pd.Series(dict_score).sort_values(ascending = False)
print ""

# Validation of the model calling the test_classifier function from tester.py
test_classifier(clf, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
