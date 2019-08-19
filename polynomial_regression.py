#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 17:20:20 2019

@author: ahmetkaanipekoren
"""

def main():
    df = pd.read_csv("kc_house_data.csv")
    
    data_x = df["sqft_living"]
    data_y = df["price"]
    
    train_x, test_x, train_y , test_y = train_test_split(data_x,data_y,test_size = 0.3 ,random_state = 42)
    
    train_x = normalization(train_x)
    test_x = normalization(test_x)
    train_y = normalization(train_y)
    test_y = normalization(test_y)
    

    return train_x, test_x, train_y , test_y


def normalization(df):
    
    df = (df - df.min()) / (df.max() - df.min())

    return df



def sorting_the_inputs(x,y):
    x = np.array(x)
    y = np.array(y)
    mapped = zip(x,y)
    mapped = sorted(mapped)
    x,y = list(zip(*mapped))
    x = np.array(x)
    y = np.array(y)
    
    return x,y




def weights(x_column,identity_matrix,ridge,y):
    
    x_x = np.linalg.inv(np.add(np.dot(x_column.T,x_column), ridge * identity_matrix))
    
    x_y = np.dot(x_column.T, y)
    
    weight = np.dot(x_x,x_y)
    
    return weight


def identity_and_input_matrix(x,degree):
    
    identity_matrix = np.identity(degree + 1)
    full_ones = np.ones(len(x))
    x_column = np.array([x,full_ones]).T
    
    temp = 2 
    
    while  temp < degree + 1:
        x_column = np.append(np.array(np.power(x,temp))[np.newaxis].T,x_column,axis=1)
        temp += 1
    
    return x_column,identity_matrix



def polynomial_regression(train_x,train_y):
    
    complexity = 1
    
    prediction_list = []
    error_list = []
    
    
    while complexity < 7 :
         x_column,identity_matrix = identity_and_input_matrix(train_x,complexity)
         weight = weights(x_column,identity_matrix,0.00001,train_y)
         
         reverse_weights = weight[::-1]
         
         y_prediction = reverse_weights[0]
         i = 1 
         while i < len(reverse_weights):
             y_prediction = y_prediction + (train_x**i) * reverse_weights[i]
             
             i += 1
             
         plt.scatter(train_x,train_y)
         plt.plot(train_x,y_prediction)
         plt.show()
         error_list.append(mse(np.array(train_y),np.array(y_prediction)))
         
         prediction_list.append(y_prediction)
         complexity += 1
        
        
    return error_list

    
    
    
def mse(y_true, y_prediction):
    
    return  np.sqrt((y_true - y_prediction)**2).mean()




def rmse_plot(error_list,color):
   
    plt.title("Root Mean Square Error of Train and Test Data")
    plt.xlabel("Complexity")
    plt.ylabel("RMSE score")
    plt.ylim([0,1])
    plt.plot(range(0,len(error_list)),error_list,color= color)
    
    
    


if __name__=="__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    
    train_x, test_x, train_y , test_y = main()
    train_x,train_y = sorting_the_inputs(train_x,train_y)
    
    error_list = polynomial_regression(train_x, train_y)
    print(error_list)
    
    

    
   
    
    
    