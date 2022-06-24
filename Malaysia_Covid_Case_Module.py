# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:00:55 2022

@author: Amirah Heng
"""

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

#%% CLASSES & FUNCTIONS

class ModelCreation():
    def __init__(self):
        pass
    
    def simple_lstm_layer(self,nb_features,num_node=32,drop_rate=0.2, 
                          output_node=1):
        '''

        Parameters
        ----------
        nb_features : int
            number of features in Input layer.
        num_node : Int, 
            number of nodes for each layer. The default is 32.
        drop_rate : float, 
            dropout rate for each lyer. The default is 0.2.
        output_node : int, optional
            number of node in output. The default is 1.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        '''
        model = Sequential()
        model.add(Input(shape=(nb_features,1)))  #(30,1)
        model.add(LSTM(num_node)) 
        model.add(Dropout(drop_rate))
        model.add(Dense(output_node,activation='relu')) 
        model.summary()    

        return model
    
class ModelAnalysis():       
    def __init__(self):
        pass
    
    def PlotHistory(self, hist):
        '''
        This is to evaluate model by plotting loss and mape 

        Parameters
        ----------
        hist : history object of tensorflow
            .

        Returns
        -------
        Graph plot of loss and mape function.

        '''
        
        hist_keys = [i for i in hist.history.keys()]
        
        plt.figure()
        plt.plot(hist.history[hist_keys[0]], label= 'loss')
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(hist.history[hist_keys[1]],label='mape')
        plt.legend()
        plt.show()
 
class ModelEvaluation():      
    def __init__(self):
        pass
    
    def plot_predicted_graph(self,test_df,predicted):
        '''
        This plots graph to compare predicted stock price 
        with actual stock price

        Parameters
        ----------
        test_df : Array
            trained dataset.
        predicted : Array
            testing dataset.
        Returns
        -------
        Graph of Actual Covid Case comparison to Predicted Covid Case .

        '''            
        plt.figure()
        plt.plot(test_df,'b',label='Actual Covid Case')
        plt.plot(predicted,'r',label='Predicted Covid Case')
        plt.legend()
        plt.show()
