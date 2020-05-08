#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/") # directory of feature_format script
sys.path.append("../graph/") # directory of graphs

from feature_format import featureFormat, targetFeatureSplit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Definition of the functions --------------------------------+

# Function to generate and save scatter plots separating poi and others 
def plot_data_points (data_x, data_y):
    plt.ioff() # Turn interactive mode off
    plt.scatter(df[data_x][df['poi'] == False], df[data_y][df['poi'] == False],
                color = 'b', label = 'poi')
    plt.scatter(df[data_x][df['poi'] == True], df[data_y][df['poi'] == True], 
                color = 'r', label = 'others')
    plt.xlabel(data_x)
    plt.ylabel(data_y)
    plt.savefig("../graph/"+data_x+"_vs_"+data_y+".png")
    plt.show() 

# Function that returns a dataframe of all people having a Z-score in the specified feature 
# not in 99.9% of population Z not in [-3.27,3.27]. POI are identified
def check_outliers(feature, z_critical):
    # Z-score of max feature
    df_feat = df[~df[feature].isin([0])][[feature, 'poi']] # only not null values
    if (len(df_feat[feature])) <= 2: # no standard deviation
        return "Not enough data"
    mu = df_feat[feature].mean() # mean population
    sig = df_feat[feature].std(ddof = 0) # standard deviation of population
    if sig == 0: # avoid zero division
        return "risk of dividing by 0 (all population equal)"
    df_feat['Z_feat'] = (df_feat[feature] - mu)/sig # Z-score column
    return df_feat[(df_feat['Z_feat'] >= z_critical) | 
                   (df_feat['Z_feat'] <= -z_critical)].sort_values('Z_feat', ascending = False)
    
# ---------- Main script ------------------------------------------------+

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Creating a dataframe from the data dictiopnnary
df = pd.DataFrame(data_dict).T  # dataframe needs to be transpose to have names as index

### Clean data and check for outliers

print "*** Check dimensions of the dataset ***"
print "Number of rows and columns:", df.shape
print "Number of POI : ", len(df[df['poi'].isin([True])])
print ""

# Converting all columns to float (all usefull columns can be converted to floats) and replace NaN by 0
df = df.apply(pd.to_numeric, errors ='coerce').fillna(0)

# Checking the number of 0 per column
print "*** Number of 0 values per columns ***"
print df.isin([0]).sum().sort_values(ascending = False).head(10)
print ""

# serie of null values per feature
ser_null = df.isin([0]).sum().sort_values(ascending = False)
ser_null.drop(['email_address', 'poi'], inplace = True) # droping email_address and poi
ser_null = ser_null / df.shape[0] * 100 # Converting to percentage
# Plotting missing information
plt.ioff() # Turn interactive mode off
ser_null.plot(kind='bar', alpha = 0.6, figsize=(16, 6))
plt.title('missing_values')
plt.xlabel("features")
plt.ylabel("proportion (%)")
plt.savefig("../graph/missing_values.png")
plt.show()
# The loan_advances has nearly no values, we won't select this feature

# Checking the number of 0 values per row
print "*** Number of 0 values per row ***"
print df.isin([0]).sum(axis = 1).sort_values(ascending = False).head(10)
print ""
# Drop the THE TRAVEL AGENCY IN THE PARK and LOCKHART EUGENE E which has no data
df.drop('THE TRAVEL AGENCY IN THE PARK', inplace = True)
df.drop('LOCKHART EUGENE E', inplace = True)

# Bonus vs salary
plot_data_points(data_x = 'salary', data_y = 'bonus')
# Check for the index of the outlier
print "Name with highest salary:", df['salary'].idxmax() # It corresponds to the TOTAL row
# Removing the TOTAL row
df.drop('TOTAL', inplace = True)
# Replot bonus vs salary
plot_data_points(data_x = 'salary', data_y = 'bonus')
print ""

# Correlation matrix (high pearson r can indicate the presence of outliers) 
print "*** Check for correlations ***"
plt.ioff() # Turn interactive mode off
plt.matshow(df.corr(method = 'pearson'))
plt.title('correlation_matrix')
plt.savefig("../graph/correlation_matrix.png")
plt.show()

### Check for seemingly correlated features:
       
# Feature 5 and feature 20
print "High Pearson r for:", list(df)[5], 'and' , list(df)[20]
plot_data_points(data_x = 'exercised_stock_options', data_y = 'total_stock_value') 
# no outlier but interresting tendency

# Feature 10 and feature 19
print "High Pearson r for:", list(df)[10], 'and' , list(df)[19]
plot_data_points(data_x = 'loan_advances', data_y = 'total_payments')
print "Name with highest total_payments:", df['total_payments'].idxmax() 
# It corresponds to the KENNETH LAY row. We have to keep it

# Feature 17 and feature 18
print "High Pearson r for:", list(df)[17], 'and' , list(df)[18]
plot_data_points(data_x = 'shared_receipt_with_poi', data_y = 'to_messages') 
# no outlier but interresting (surprising) tendency

print "*** Re-check dimensions of the dataset ***"
print "Number of rows and columns:", df.shape
print "Number of POI : ", len(df[df['poi'].isin([True])])

### Check for outliers by feature out of 99% population

# returns all people not in 99.9% of population for all features
for feature in [f for f in list(df) if f != 'poi']:
    print check_outliers(feature,  3.27)
    print ""

""" Identified not POI outliers:
| LAVORATO JOHN J | Bonus | Z = 4.74 |
| LAVORATO JOHN J | from_poi_to_this_person | Z = 5.09 |
| FREVERT MARK A | deferral_payments | Z = 4.39 |
| FREVERT MARK A | other | Z = 5.01 |
| FREVERT MARK A | salary | Z = 4.41 |
| KAMINSKI WINCENTY J | from_messages | Z = 7.52 |
| MARTIN AMANDA K | long_term_incentive | Z = 5.14 |
| WHITE JR THOMAS E | restricted_stock | Z = 5.67 |
| BHATNAGAR SANJAY | restricted_stock_deferred | Z = 3.98 |
| SHAPIRO RICHARD S | to_messages | Z = 5.09 |
"""

### Tasks to perform in poi_id.py:
"""
- Convert all columns to numeric and fill the NaN by 0
- Drop TOTAL row
- Drop THE TRAVEL AGENCY IN THE PARK row
- Drop LOCKHART EUGENE E, LAVORATO JOHN J, FREVERT MARK, KAMINSKI WINCENTY J, MARTIN AMANDA K, WHITE JR THOMAS E, BHATNAGAR SANJAY, SHAPIRO RICHARD S
- Features email_adresses and loan_advances not to be used
"""

