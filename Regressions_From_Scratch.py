#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Aleksander Merrill
Functions for Project 4 - Logistic, Linear, and NN regression/classification
"""
import numpy as np
import pandas as pd


def mean_err(real, predicted):
    """
    Returns Mean Squared Error of two series
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
    return error_sq


# In[2]:


def log_reg(train, test, rate, error = .05, iterations = 1000):
    """
    Logistic regression classifier (1 /(1 + exp(-X*Beta)))
    @param train - the input data to train on
    @param error - the error tolerance to stop iterating
    @param rate - learning rate
    """
    show_weights = 1
    xtrain = train.drop(['class'], axis = 1, inplace = False)
    ytrain = train['class']
    xtest = test.drop(['class'], axis = 1, inplace = False)
    ytest = test['class']
    yhat = []
    y_classes = ytrain.unique()
    y_classes = pd.Series(y_classes).sort_values(ascending = True)
    y_class_max = y_classes.max()
    wx = 0
    Beta = pd.Series(np.zeros(xtrain.shape[1]))

    for i in range(iterations):
        Beta.index = xtrain.columns
        wx = np.dot(Beta.astype(np.float), xtrain.transpose())
        P = 1/(1 + np.exp(-(wx)))
        Q = P - ytrain
        Q.index = xtrain.index
        dBeta = xtrain.transpose().dot(Q)
        
        if show_weights == 1:
            print("Weights")
            print(Beta)
            print("Gradient:")
            print(dBeta)
            print("Updated Beta:")
            print(Beta - rate * dBeta)
            show_weights = 0
            
        
        Beta = Beta - rate * dBeta
        wx = 0
        if np.linalg.norm(dBeta.values) < error:
            break
            
    Beta.index = xtest.columns
    yhat = xtest.dot(Beta)
    for i in range(len(yhat)):
        if yhat[i] > y_class_max:
            yhat[i] = y_class_max
            continue
        for y_class in y_classes:
            if yhat[i] <= y_class + .5:
                yhat[i] = y_class
                break  
    return yhat, ytest


# In[3]:


def lin_reg(train, test, learn_rate, error = .05, iterations = 1000):
    """
    Linear regression for regression problems, returns coefficient vector for regression problem
    @param train - training data
    @param error - error tolerance to stop iterating
    @param iterations - max number of times algorithm will run
    """
    train.reset_index()
    test.reset_index()
    
    x = train.drop('class', axis = 1, inplace = False)
    xtest = test.drop('class', axis =1, inplace = False)
    xtest['constant'] = np.ones(xtest.shape[0])
    ytest = test['class']
    x['constant'] = np.ones(x.shape[0])
    y = train['class']
    weights = pd.Series(np.ones(x.shape[1]))
    weights.index = x.columns
    for i in range(iterations):
        ys = pd.Series(np.ones(x.shape[0]))
        ys.index = x.index
        for idx, row in x.iterrows():
            ys[idx] = weights.dot(row)
            
        if mean_err(ys, y) < error:
            break
        xt = x.transpose()
        dweights = 2 * xt.dot(y - ys)/x.shape[0]
        weights = weights.add(dweights.multiply(other = 0 - learn_rate))
        weights.index = x.columns
    
    y_hat = []
    for ind, rowtest in xtest.iterrows():
        y_i = weights.dot(rowtest)
        y_hat.append(y_i)
    return y_hat, ytest


# In[ ]:


def sigmoid(x):
    """
    take sigmoid function on information x
    @param x - input value
    """
    return 1/(1 + np.exp(-x))


# In[1]:


def sig_prime(x):
    """
    derivative of sigmoid function
    @param x - input value
    """
    return x * (1 - x)


# In[ ]:





# In[ ]:



def back_prop(train, test, alg_type, learn_rate, iterations = 1000):
    """
    Back propagation algorithm
    @param train - training data
    @param alg_type - classification or regression
    @param learn_rate - learning rate
    @param iterations - max number of times algorithm will run
    """
    show_weights = 1
    
    x = train.drop(['class'], axis=1, inplace = False)
    y = train['class']
    y = y.values.reshape((x.shape[0], 1))
    x_test = test.drop(['class'], axis =1, inplace = False)
    y_test = test['class']
    is_log = (alg_type == 'classification')
    inw = pd.DataFrame(np.random.uniform(size = (x.shape[1], 5)))
    in2w = pd.DataFrame(np.random.uniform(size = (5, 5)))
    outw = pd.DataFrame(np.random.uniform(size = (5, 1)))
    for i in range(iterations):
        #inw.index = x.columns
        hid1 = np.dot(x, inw)
        hid1_sig = sigmoid(hid1.astype(np.float))
        hid2 = np.dot(hid1_sig, in2w)
        hid2_sig = sigmoid(hid2.astype(np.float))
        out = np.dot(hid2_sig, outw)
        output = sigmoid(out.astype(np.float))
        
        out_err = y - output
        out_gradient = sig_prime(output.astype(np.float))
        out_change = (out_err * out_gradient * learn_rate).reshape(x.shape[0], 1)
        if show_weights == 1:
            print("Error of output:")
            print(out_err)
            print("Output gradient:")
            print(out_gradient)
            print("Change in output weight:")
            print(out_change)
        
        hid2_err = np.dot(out_change, outw.transpose())
        hid2_gradient = sig_prime(hid2_sig.astype(np.float))
        hid2_change = hid2_err * hid2_gradient * learn_rate
        if show_weights == 1:
            print("Error of Second Layer:")
            print(hid2_err)
            print("Second Layer gradient:")
            print(hid2_gradient)
            print("Change in Second Layer weight:")
            print(hid2_change)
        
        hid1_err = np.dot(hid2_change, in2w.transpose())
        hid1_gradient = sig_prime(hid1_sig.astype(np.float))
        hid1_change = hid1_err * hid1_gradient * learn_rate
        if show_weights == 1:
            print("Error of Second Layer:")
            print(hid1_err)
            print("Second Layer gradient:")
            print(hid1_gradient)
            print("Change in Second Layer weight:")
            print(hid1_change)
            show_weights = 0
        
        outw += np.dot(hid2_sig.transpose(), out_change)
        in2w += np.dot(hid1_sig.transpose(), hid2_change)
        inw += np.dot(x.T, hid1_change)
    
    hid1 = np.dot(x_test, inw)
    hid1_sig = sigmoid(hid1.astype(np.float))
    hid2 = np.dot(hid1_sig, in2w)
    hid2_sig = sigmoid(hid2.astype(np.float))
    out = np.dot(hid2_sig, outw)
    yhat = sigmoid(out.astype(np.float))
    hatmean = train['class'].mean()
    yhat = np.nan_to_num(yhat, nan=hatmean)
    if is_log:
        y_classes = y_test.unique()
        y_classes = pd.Series(y_classes).sort_values(ascending = True)
        y_class_max = y_classes.max()
        print(y_class_max)
        for i in range(len(yhat)):
            if yhat[i] > y_class_max:
                yhat[i] = y_class_max
                continue
            for j in range(len(y_classes)):
                if yhat[i] <= y_classes[j] + .5:
                    yhat[i] = y_classes[j]
                    break
    yhat = pd.Series(np.squeeze(yhat))
    print(yhat[0])
    return yhat, y_test


# In[1]:


def encode(x, columns, threshold):
    """
    Encode step of autoencoder, remove feature with lowest Standard Deviation and append to list of removed columns
    This is chosen for feature selection due to standardization of all features
    @param x - the input features x
    @param columns - the columns already removed
    @param threshold - the minimum StdDev allowed
    @return x_2 - the reduced features x
    @return columns - the removed column names and columns in a list of tuples
    """
    min_feature = ""
    min_std = 10000
    x_2 = x.copy()
    for feature in x.columns:
        x_std = x[feature].std()
        if x_std < min_std:
            min_std = x_std
            min_feature = feature
    if threshold > min_std:
        columns.append((min_feature, x[min_feature]))
        x_2[min_feature][:] = 0
    return (columns, x_2)
    
def encodeTest(x, columns):
    """
    Encodes the test features according to trained column set
    @param x - the test data features
    @param columns - the learned encoded features and values
    @return x2 - encoded test data
    """
    x2 = x.copy()
    for (feature, column) in columns:
        x2[feature][:] = 0
    return x2
    


# In[ ]:


def decode(x, columns):
    """
    Decode step of autoencoder
    @param x - the reduced features x
    @param columns - the removed columns being added back in; list of (feature, values) tuple
    @return x_hat - the 'original input'
    @return columns - the columns that have been removed so far
    """
    x_hat = x.copy()
    for (feature, column) in columns:
        x_hat[feature] = column
    x_hat = pd.DataFrame(x_hat)
    return (columns, x_hat)
    


# In[ ]:


def auto_encode(train, test, alg_type, learn_rate, threshold = 1, iterations = 1000):
    """
    Autoencoder algorithm
    @param train - training data
    @param alg_type - classification or regression
    @param learn_rate - learning rate
    @param iterations - max number of times algorithm will run
    @param threshold - the threshold of standard deviation for encoding
    
    NOTE:Input --> Encode/Reduction -> Hidden layer training w/reduced features -> Decode -> Update Weights 
    """
    X = train.drop(['class'], axis = 1, inplace = False)
    x = X.copy()
    y = train['class'].values.reshape(x.shape[0], 1)
    x_test = test.drop(['class'], axis = 1, inplace = False)
    y_test = test['class']
    columns = []
    norm = 1000000
    is_log = (alg_type == 'classification') #Check if is classification problem
    inw = np.random.uniform(size = (x.shape[1], 5))
    outw = np.random.uniform(size = (5, 1))
    for i in range(iterations):
        (new_cols, x_2) = encode(x, columns, threshold)
        hid1 = np.dot(x_2, inw)
        hid1_sig = sigmoid(hid1.astype(np.float))
        out = np.dot(hid1_sig, outw)
        output = sigmoid(out.astype(np.float))
        (new_cols, x_hat) = decode(x_2, new_cols)
        Q = X - x_hat
        if np.linalg.norm(Q) < norm:
            columns = new_cols
            x = x_hat
        else:
            break
            
        
        
        
        out_err = y - output
        out_gradient = sig_prime(output.astype(np.float))
        out_change = out_err * out_gradient * learn_rate
        
        hid1_err = np.dot(out_change, outw.transpose())
        hid1_gradient = sig_prime(hid1_sig.astype(np.float))
        hid1_change = hid1_err * hid1_gradient * learn_rate
        
        outw = outw + np.dot(hid1_sig.transpose(), out_change)
        inw = inw + np.dot(x.transpose(), hid1_change) * learn_rate
    x_test_2 = encodeTest(x_test, columns)
    hid1 = np.dot(x_test_2, inw)
    hid1_sig = sigmoid(hid1.astype(np.float))
    out_t = np.dot(hid1_sig, outw)
    yhat = sigmoid(out_t.astype(np.float))
    
    if is_log:
        y_classes = y_test.unique()
        y_classes = pd.Series(y_classes).sort_values(ascending = True)
        y_class_max = y_classes.max()
        print(y_class_max)
        for i in range(len(yhat)):
            if yhat[i] > y_class_max:
                yhat[i] = y_class_max
                continue
            for j in range(len(y_classes)):
                if yhat[i] <= y_classes[j] + .5:
                    yhat[i] = y_classes[j]
                    break
    yhat = pd.Series(np.squeeze(yhat))
    
    return yhat, y_test

