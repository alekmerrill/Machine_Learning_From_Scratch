#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Aleksander Merrill
Introduction to Machine Learning 605.649
"""

import numpy as np
import pandas as pd
import math

def mode(array):
    """
    Find mode of a list
    """


# In[2]:


def distance(point1, point2):
    """
    A function to find the distance between two points, uses Euclidean for ease
    @param - point1 - the first point; has d number of dimensions; a row in a dataframe
    @param - point2 - the second point; same dimensions as point1; a row in a dataframe
    @return - distance between point1 and point2
    """
    distance = 0
    if not len(point1) == len(point2):
        raise Exception("Points do not have the same dimensions")
    for index in range(len(point1)):
        distance += (point1[index] - point2[index])**2
    distance = np.sqrt(distance)
    return distance


# In[3]:

def gaussian(data, k):
    """
    Calculate Gaussian Kernel
    @param data - the data being smoothed
    @param k - the number of values
    @param sigma - sigma of Gaussian kernel equation, calculated as standard dev of data 
    @return smoothed value
    """
    sigma = np.std(np.array(data))
    if sigma == 0: #to ensure no divide by 0 errors
        sigma = .00001
    coeff = (np.sqrt(np.pi * 2)*sigma)**k
    coeff = 1 / coeff
    expon = - (np.linalg.norm(data)**2)/(2 * sigma**2)
    exp = np.exp(expon)
    return coeff * exp


def KNN(train, test, k, style):
    """
    Using training data, classify the testx data using K-Nearest Neighbors
    Uses the Mean for the 
    @param - train - the train data
    @param - test - the test data
    @param - k - the number of neighbors to compare with
    @param - style - classification or regression
    @return - the predictions for each value in the test set
    """
    printing_gauss = 0
    printing_class = 0
    predictions = []
    trainy = train['class']
    true_class = test['class']
    trainx = train.loc[:, train.columns != 'class']
    testx = test.loc[:, test.columns != 'class']
    if k > len(true_class) - 1:
        raise Exception("K is larger than the test set, choose a smaller K")
    for testdex, testrow in testx.iterrows():
        k_nearest = [] #list of row information
        k_nearest_dist = [] #list of distances
        for traindex, trainrow in trainx.iterrows():
            dist = distance(trainrow, testrow)
            if traindex < k:
                k_nearest.append(trainy[traindex])
                k_nearest_dist.append(dist)
                k_nearest = [i for _,i in sorted(zip(k_nearest_dist,k_nearest))]
                k_nearest_dist.sort()
                continue
            else:
                if dist < k_nearest_dist[-1]:
                    k_nearest[-1] = trainy[traindex]
                    k_nearest_dist[-1] = dist
            k_nearest = [i for _,i in sorted(zip(k_nearest_dist,k_nearest))]
            k_nearest_dist.sort()
        if style == 'classification': #Classification
            predict = pd.Series(k_nearest).mode()[0]
            if printing_class == 0:
                print("Current point:")
                print(testrow)
                print("K neighbors Classifications:")
                print(k_nearest)
                printing_class = 1
                print("Prediction: ")
                print(predict)
            predictions.append(predict)
        else:      #Regression           
            predict = gaussian(k_nearest, k)
            if printing_gauss == 0:
                print("Current point:")
                print(testrow)
                print("K neighbors")
                print(k_nearest)
                print("Prediction: ")
                print(predict)
                printing_gauss = 1  
            predictions.append(predict)
    predictions = pd.Series(predictions)
    return (true_class, predictions)


# In[ ]:


def condensedNearestNeighbor(train, test, style, error = .05, k = 5):
    """
    Create a condensed training set Z from the set X, as labeled in the lecture
    notes
    @param train - the original training set
    @param test - the testing set
    @param style - regression or classification
    @param error - the possible error allowed for regression class; .05 default
    @param Z - the condensed training set that will be used for k = tuned k KNN
    @return KNN where k = 1 on Z and test
    """
    edit_example = 0
    Z = pd.DataFrame(columns = train.columns) #intitialize empty set of training data
    X = train.copy()
    X_x = X.loc[:, X.columns != 'class']
    #X set to have values removed; this way train can be used later as well
    for xindex, xrow in X_x.iterrows():
        if len(Z) == 0:
            Z = Z.append(X.iloc[xindex])
            X = X.drop(index = xindex)
            continue
        Z_x = Z.loc[:, Z.columns != 'class']
        minimum_dist = Z_x.iloc[0] #initialize the smallest distance row in Z
        dist = distance(xrow, minimum_dist)
        distdex = 0
        for zindex, zrow in Z.iterrows():
            dist2 = distance(xrow, Z_x.loc[zindex])
            if dist2 < dist:
                dist = dist2
                distdex = zindex
        if style == 'regression':
            xvalue = X.at[xindex, 'class']
            #divide by 0 handling
            if xvalue == 0:
                xvalue = .0001
            real_error = (Z.at[distdex, 'class'] - X.at[xindex, 'class'])
            real_error = real_error / xvalue
            if real_error > error:
                Z = Z.append(X.loc[xindex])
                X = X.drop(xindex)
            else:
                continue
        else:
            if not Z.at[distdex, 'class'] == X.at[xindex, 'class']:
                #Additon of row to Z, demonstrate its addition with updated shape
                if edit_example == 0:
                    print("Original Z shape:")
                    print(Z.shape)
                Z = Z.append(X.loc[xindex])
                if edit_example == 0:
                    print("New Z:")
                    print(Z.shape)
                    edit_example = 1
                X = X.drop(index = xindex)
            else:
                continue
    return KNN(Z, test, k, style)
            
def ENN(train, test, style, error, k = 5):
    """
    Implementing Edited Nearest Neighbor
    @param train - the training data
    @param test - the testing data
    @param style - classification or regression
    @param error - how incorrect a regression can be to be considered "wrong"
    @return KNN with tuned K
    """
    edit_example = 0
    X = train.copy()
    train_x = train.loc[:, train.columns != 'class']
    X_x = train_x.copy()
    #X set to have values removed; this way train can be used later as well
    for traindex, trainrow in train_x.iterrows():
        k_nearest = [] #initialize list of train row indices
        k_nearest_dist = [] #training row distances from location
        for xindex, xrow in X_x.iterrows():
            if not xindex in X.index: #used to prevent index not in X indices
                continue
            dist = distance(trainrow, xrow)
            if len(k_nearest) < k:
                k_nearest.append(xindex)
                k_nearest_dist.append(dist)
                k_nearest = [i for _,i in sorted(zip(k_nearest_dist,k_nearest))]
                k_nearest_dist.sort()
                continue
            else:
                if len(k_nearest_dist) < 1:
                    break
                if dist < k_nearest_dist[-1]:
                    k_nearest[-1] = xindex
                    k_nearest_dist[-1] = dist
            k_nearest = [i for _,i in sorted(zip(k_nearest_dist,k_nearest))]
            k_nearest_dist.sort()
        if style == 'regression':
            for item in k_nearest:
                xvalue = train.at[item, 'class']
                if xvalue == 0:
                    xvalue = .0001
                real_error = (train.at[item, 'class'] - train.at[traindex, 'class'])
                real_error = real_error / xvalue
                if real_error > error:
                    #Remove row from X, demonstrate its removal with updated shape
                    if edit_example == 0:
                        print("Original X shape:")
                        print(X.shape)
                    X = X.drop(index = item)
                    if edit_example == 0:
                        print("New X shape:")
                        print(X.shape)
                        edit_example = 1
        else:
            for item in k_nearest:
                if not train.at[item,'class'] == train.at[item, 'class']:
                    #Remove row from X, demonstrate its removal with updated shape
                    if edit_example == 0:
                        print("Original X shape:")
                        print(X.shape)
                    X = X.drop(index = item)
                    if edit_example == 0:
                        print("New X shape:")
                        print(X.shape)
                        edit_example = 1
    return KNN(X, test, k, style)