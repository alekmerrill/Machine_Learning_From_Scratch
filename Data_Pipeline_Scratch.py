#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Aleksander Merrill
Introduction to Machine Learning
Project 1
"""

import numpy as np
import pandas as pd
import math


# In[2]:


def load_csv(filename):
    """
    Loads data from a csv file into a pandas dataframe, renames columns
    based on information found in corresponding .names file
    Changes all predicted features to be titled 'class' for ease later
    @parameter - filename - the name of the csv file being loaded
    """
    #Change predicted class name to "Class"        
    
    
    if filename == "abalone.data":
        #Rings renamed to Class
        columns = ["Sex", "Length", "Diameter",
                  "Height", "Whole Weight", "Shucked Weight",
                  "Viscera Weight", "Shell Weight", "class"]
        df =  pd.read_csv(filename, header = None)
        df.columns = columns
    elif filename == "breast-cancer-wisconsin.data":
        columns = ["Sample Code Number", "Clump Thickness", "Uniformity of Cell Size",
                  "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
                  "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "class"]
        df =  pd.read_csv(filename, header = None)
        df.columns = columns
    elif filename == "car.data":
        #No rename required
        columns = ["buying", "maint", "doors",
                  "persons", "lugboot", "safety", "class"]
        df =  pd.read_csv(filename, header = None)
        df.columns = columns
    elif filename == "forestfires.data":
        #area renamed to class
        df =  pd.read_csv(filename)
        """df = df.rename(columns = {'area':'class'})
        is not working in import, but works when written in other file.
        Will determine why in future tests"""
    elif filename == "house-votes-84.data":
        columns = ["class", "handicapped-infants", "water-project-cost-sharing",
                  "adoption-of-the-budget-resolution", "physician-fee-freeze",
                  "el-salvador-aid", "religious-groups-in-schools",
                  "anti-satellite-test-ban", "aid-to-nicaraguan-contras",
                  "mx-missile", "immigration", "synfuels-corporation-cutback",
                  "education-spending", "superfund-right-to-sue",
                  "crime", "duty-free-exports", "export-administration-act-south-africa"]
        df =  pd.read_csv(filename, header = None)
        df.columns = columns
    elif filename == "machine.data":
        #PRP renamed to class
        columns = ["vendor name", "Model Name", "MYCT",
                  "MMIN", "MMAX", "CACH", "CHMIN",
                  "CHMAX", "class", "ERP"]
        df =  pd.read_csv(filename, header = None)
        df.columns = columns
    else:
        df = pd.read_csv(filename, header = None)

    return df


# In[5]:


def impute(data):
    """
    A function designed to impute missing values with the average of a column
    Should not be used on dataframes based on house-votes-84.data,
    as missing values are part of the data, being a "non vote" instead
    @param - data - the dataframe being imputed
    @return updated data
    """
    notAValue = 0
    for column in data.columns:
        if not (data[column].isin(['?']).any()):
            continue
        if (type(data[column][0]) is str) & data[column][0].isnumeric():
            data[column] = pd.to_numeric(data[column], errors='coerce').astype(float)
            #data[column].mask(data[column] == '?', np.nan, inplace=True)
            data[column].fillna(data[column].mean())
    return data

# In[7]:


def convertOrdinal(data):
    """
    Converts Ordinal Data to encoded numbers
    Abalone: N/A
    breast-cancer-wisconsin: N/A
    car: buying, maint, lugboot, safety
    forestfires: N/A
    house-votes-84: N/A
    machine: N/A
    @param - data - the dataframe whose Ordinal data is being converted
    @return updated data
    """
    #check if is car (does lug_boot exist?) --> assign ordinal values
    if "lugboot" in data.columns:
        data['buying'] = data['buying'].map({'vhigh':3, 'high':2, 'med':1, 'low':0})
        data['buying'] = pd.to_numeric(data['buying'], errors='coerce').astype(float)
        data.maint = data['maint'].replace({'vhigh':3, 'high':2, 'med':1, 'low':0})
        data['maint'] = pd.to_numeric(data['maint'], errors='coerce').astype(float)
        data.lugboot = data['lugboot'].replace({'small':0, 'med':1, 'big':2})
        data['lugboot'] = pd.to_numeric(data['lugboot'], errors='coerce').astype(float)
        data.safety = data['safety'].replace({'low':0, 'med':1, 'high':2})
        data['safety'] = pd.to_numeric(data['safety'], errors='coerce').astype(float)
        data.doors = data['doors'].replace({'2':2, '3':3, '4':4, '5more':5})
        data['doors'] = pd.to_numeric(data['doors'], errors='coerce').astype(float)
        data.persons = data['persons'].replace({'2':2, '4':4, 'more':6})
        data['persons'] = pd.to_numeric(data['persons'], errors='coerce').astype(float)
        return data
    #else return data as is
    else:
        return data
    


# In[8]:


def convertNominal(data):
    """
    Converts Nominal Data to dummies, deletes original column
    Abalone: Sex
    breast-cancer-wisconsin: N/A, as class is what is predicted, rest is not nominal
    car: N/A
    forestfires: month, day
    house-votes-84: Every single column but party(Yes, No, Maybe - on votes;=)
    machine: vendor name, Model Name
    @param - data - the dataframe whose Ordinal data is being converted
    @return updated data
    """
    columns = data.columns
    #If FFMC --> forestfires
    if 'FFMC' in columns:
        mon = pd.DataFrame(pd.get_dummies(data['month'], prefix = 'month'))
        data = data.join(mon)
        data.drop(['month'], axis =1, inplace = True)
        day = pd.DataFrame(pd.get_dummies(data['day'], prefix = 'day'))
        data = data.join(day)
        data.drop(['day'], axis =1, inplace = True)
    #elif Sex --> abalone
    elif 'Sex' in columns:
        sexind = pd.DataFrame(pd.get_dummies(data['Sex'], prefix = 'Sex'))
        data = data.join(sexind)
        data.drop(['Sex'], axis =1, inplace = True)
    #elif Party --> house-votes-84
    elif 'mx-missile' in columns:
        inf = pd.DataFrame(pd.get_dummies(data['handicapped-infants'], prefix = 'infant vote'))
        data = data.join(inf)
        data.drop(['handicapped-infants'], axis =1, inplace = True)
        
        water = pd.DataFrame(pd.get_dummies(data['water-project-cost-sharing'], prefix = 'water vote'))
        data = data.join(water)
        data.drop(['water-project-cost-sharing'], axis =1, inplace = True)
        
        budget = pd.DataFrame(pd.get_dummies(data['adoption-of-the-budget-resolution'], prefix = 'budget vote'))
        data = data.join(budget)
        data.drop(['adoption-of-the-budget-resolution'], axis =1, inplace = True)
        
        phys = pd.DataFrame(pd.get_dummies(data['physician-fee-freeze'], prefix = 'physician fee vote'))
        data = data.join(phys)
        data.drop(['physician-fee-freeze'], axis =1, inplace = True)
        
        elsalvador = pd.DataFrame(pd.get_dummies(data['el-salvador-aid'], prefix = 'el salvador aid vote'))
        data = data.join(elsalvador)
        data.drop(['el-salvador-aid'], axis =1, inplace = True)
        
        religious = pd.DataFrame(pd.get_dummies(data['religious-groups-in-schools'], prefix = 'religious groups vote'))
        data = data.join(religious)
        data.drop(['religious-groups-in-schools'], axis =1, inplace = True)
        
        satellite = pd.DataFrame(pd.get_dummies(data['anti-satellite-test-ban'], prefix = 'satellite test ban vote'))
        data = data.join(satellite)
        data.drop(['anti-satellite-test-ban'], axis =1, inplace = True)
        
        contra = pd.DataFrame(pd.get_dummies(data['aid-to-nicaraguan-contras'], prefix = 'contra aid vote'))
        data = data.join(contra)
        data.drop(['aid-to-nicaraguan-contras'], axis =1, inplace = True)
        
        mx = pd.DataFrame(pd.get_dummies(data['mx-missile'], prefix = 'missile vote'))
        data = data.join(mx)
        data.drop(['mx-missile'], axis =1, inplace = True)
        
        immigration = pd.DataFrame(pd.get_dummies(data['immigration'], prefix = 'immigration vote'))
        data = data.join(immigration)
        data.drop(['immigration'], axis =1, inplace = True)
        
        synfuel = pd.DataFrame(pd.get_dummies(data['synfuels-corporation-cutback'],
                                              prefix = 'synfuels cutback vote'))
        data = data.join(synfuel)
        data.drop(['synfuels-corporation-cutback'], axis =1, inplace = True)
        
        edu = pd.DataFrame(pd.get_dummies(data['education-spending'], prefix = 'edu spending vote'))
        data = data.join(edu)
        data.drop(['education-spending'], axis =1, inplace = True)
        
        superfund = pd.DataFrame(pd.get_dummies(data['superfund-right-to-sue'], prefix = 'superfund sue vote'))
        data = data.join(superfund)
        data.drop(['superfund-right-to-sue'], axis =1, inplace = True)
        
        crime = pd.DataFrame(pd.get_dummies(data['crime'], prefix = 'crime vote'))
        data = data.join(crime)
        data.drop(['crime'], axis =1, inplace = True)
        
        duty = pd.DataFrame(pd.get_dummies(data['duty-free-exports'], prefix = 'duty free vote'))
        data = data.join(duty)
        data.drop(['duty-free-exports'], axis =1, inplace = True)
        
        southafr = pd.DataFrame(pd.get_dummies(data['export-administration-act-south-africa'],
                                               prefix = 'south africa export vote'))
        data = data.join(southafr)
        data.drop(['export-administration-act-south-africa'], axis =1, inplace = True)
    #elif MMAX --> machine
    elif 'MMAX' in columns:
        vend = pd.DataFrame(pd.get_dummies(data['vendor name'], prefix = 'vendor'))
        mod = pd.DataFrame(pd.get_dummies(data['Model Name'], prefix = 'Model'))
        data = data.join(vend)
        data = data.join(mod)
        data.drop(['vendor name', 'Model Name'], axis = 1, inplace = True)
    return data


# In[9]:


def discretization(data, bins, style):
    """
    A function to discretize data in a dataframe with an input number of bins
    in either the equal-width or equal-frequency style
    @param - data - the data being discretized
    @param - bins - the number of bins desired
    @param - style - equal-width or equal-frequency; equal-frequency is assumed for if-else
    @return - discretized data
    """
    if style == 'equal-width':
    #equal-width: width = range/bins for all columns
        for column in data.columns:
            if '_' in column or ('class' == column and type(data['class'][0]) == str):
                continue
            colmin = data[column].min()
            colmax = data[column].max()
            width = math.floor((colmax - colmin) / bins)
            bindex = 0
            colind = 0
            binmax = colmin
            binmaxes = []
            while bindex < bins:
                binmax = binmax + width
                binmaxes.append(binmax)
                bindex += 1
            while colind < len(data.index):
                value = data.at[colind, column]
                for binval in binmaxes:
                    if value > max(binmaxes):
                        data.at[colind, column] = colmax
                    elif value < binval:
                        data.at[colind, column] = binval
                colind += 1
            
    else:
    #equal-frequency assumed
        for column in data.columns:
            if '_' in column or ('class' == column and type(data[column][0]) == str):
                continue
            numInBins = math.floor(len(data[column]) / bins)
            data.sort_values(by = column, ascending = False)
            colindex = 0
            while colindex < len(data[column]):
                value = data[column][colindex]
                for bindex in range(numInBins):
                    if colindex >= len(data[column]):
                        break
                    data.at[colindex, column] = value
                    colindex = colindex + 1
    return data


# In[10]:


def standardize(train, test):
    """
    Performs z-score standardization on training and test set
    Test set uses mean and stddev of training set in standardization
    @param - train - the training data
    @param - test - the test data
    @return - train, test data standardized
    """
    # Change this so it only standardizes numerical values
    if not (train.columns.all() == test.columns.all()):
        raise Exception("Train and test data must have same features")
    for column in train.columns:
        if type(train.at[0, column]) == str and not train.at[0, column].isnumeric():
            continue
        if '_' in column:
            continue
        mean = train[column].mean()
        stddev = train[column].std()
        traindex = 0
        while traindex < len(train.index):
            x = train.at[traindex, column]
            zscore = (x - mean) / stddev
            train.at[traindex, column] = zscore
            traindex = traindex + 1
        testdex = 0
        while testdex < len(test.index):
            y = test.at[testdex, column]
            zscore = (y-mean)/stddev
            test.at[testdex, column] = zscore
            testdex += 1
    return train, test


# In[11]:


def kfoldCross(data, k=5):
    """
    K Fold cross validation
    Source: Author- Aleksander Merrill; Assignment - Algorithms for Data Science HW3
    @param - data - the data being separated
    @param - k - the number of times the split will be done, assumes k = 5 unless specified
    @return - ReturnSets - the k train-test sets
    """
    count = math.floor(data.shape[0]/k)
    trainSets = []
    testSets = []
    for j in range(k):
        test = data[count * j: count*(j + 1)]
        train = data[0:count*j]
        train = train.append(data[count*(j+1):])
        train.reset_index(drop = True, inplace = True)
        test.reset_index(drop = True, inplace = True)
        trainSets.append(train)
        testSets.append(test)
    ReturnSets = (trainSets, testSets)
    return ReturnSets


# In[12]:


def classificationEval(real, predicted):
    """
    Returns classification score for classification tasks
    @param real - the real classifications
    @param predicted - the prediction classifications
    @return accuracy score
    """
    correct = 0
    prediction_count= len(real)
    pred = 0
    compares = (real == predicted)
    while pred < prediction_count:
        if compares[pred]:
            correct += 1
        pred += 1
    accuracy = correct / prediction_count
    print("Classification accuracy: " + str(accuracy))
    return accuracy


# In[14]:


def regressionEval(real, predicted):
    """
    Returns Mean Squared Error of prediction vs real values
    @param real - the real value classifications
    @param predicted - the prediction value classifications
    @return mean squared error
    """
    prediction_count = len(real)
    error_sq = 0
    for pred in range(prediction_count):
        error = np.square(real[pred] - predicted[pred])
        error_sq += error
    error_sq = (1 / prediction_count) * error_sq
    print("Mean squared error: " + str(error_sq))
    return error_sq


# In[15]:


def majorityPredictor(train, test, pred_type):
    """
    A naive majority predictor, returns the real
    and predicted values. The predicted values are as follows:
    classifier problem - majority predictor
    regression problem - average predictor
    @param - train - the training set
    @param - test - the test set
    @pred_type - the type of prediction, regression or classifier
    """
    predictions = []
    real = pd.Series(test['class'])
    pred_count = len(test['class'])
    if pred_type == 'regression':
    #Return regression predictor
        prediction = train['class'].mean()
        predictions = [prediction] * pred_count
        predictions = pd.Series(predictions)
    else:
    #Return classifier predictor
        prediction = train['class'].mode()[0]
        predictions = [prediction] * pred_count
        predictions = pd.Series(predictions)
    return predictions, real

