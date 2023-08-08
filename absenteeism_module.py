#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


class AbsenteeismModel:
    
    def __init__(self):
        with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
        
        self.data = None
        
    # take a data file (*.csv) and preprocess it in the same way as in the lectures
    def load_and_preprocessed_data(self, data_file):
        # import the data
        df = pd.read_csv(data_file, sep=',')
        
        # store the data in a new variable for later use
        self.raw_df = df.copy()
        
        # drop the 'ID' column
        df = df.drop(columns = ['ID'])
        
        # We are adding columns named "Absenteeism Time in Hours" with "NaN" values in this dataset so that 
        # we can just copy the commands from preprocessing code and paste here without major changes
        df['Absenteeism Time in Hours'] = "NaN"
        
        # create a separate dataframe, containing dummy values for ALL avaiable reasons
        reason_column = pd.get_dummies(df['Reason for Absence'], drop_first=True)
        
        # to avoid multicollinearity, drop the 'Reason for Absence' column from df
        df = df.drop('Reason for Absence', axis=1)
        
        # split reason_columns into 4 types
        reason_type_1 = reason_column.loc[:,'1':'14'].max(axis=1)
        reason_type_2 = reason_column.loc[:,'15':'17'].max(axis=1)
        reason_type_3 = reason_column.loc[:,'18':'21'].max(axis=1)
        reason_type_4 = reason_column.loc[:,'22':'28'].max(axis=1)
        
        # to create 4 different columns for 4 differnt reasons category in df
        df['Reason_1'] = reason_type_1
        df['Reason_2'] = reason_type_2
        df['Reason_3'] = reason_type_3
        df['Reason_4'] = reason_type_4
        
        # re-order the columns in df
        columns_1st_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 
                                 'Transportation Expense', 'Distance to Work', 'Age', 
                                 'Daily Work Load Average', 'Body Mass Index', 'Education',
                                 'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[columns_1st_reordered]
        
        # convert the 'Date' column into datetime
        df['Date'] = pd.to_datetime(df['Date'], format= '%d/%m/%Y')
        
        # create a list with month values retrieved from the 'Date' column
        list_months = []
        for i in range(len(df['Date'])):
            list_months.append(df['Date'][i].month)
            
        # insert the values in a new column in df, called 'Month Value'
        df['Month'] = list_months
        
        # create a new feature called 'Day of the Week'
        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())
        
        # drop the 'Date' column from df
        df = df.drop(['Date'], axis = 1)
        
        # re-order the columns in df
        columns_2nd_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month',
                               'Day of the Week', 'Transportation Expense', 'Distance to Work', 'Age',
                               'Daily Work Load Average', 'Body Mass Index', 'Education',
                               'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[columns_2nd_reordered]
        
        # map 'Education' variables; the result is a dummy
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        
        # replace the NaN values
        df = df.fillna(value=0)
        
        # drop the original absenteeism time
        df = df.drop(['Absenteeism Time in Hours'],axis=1)
        
        # drop the variables we decide we don't need
        df = df.drop(['Day of the Week','Daily Work Load Average','Distance to Work'],axis=1)
        
        # store the preprocessed data into new variable for later use
        self.preprocessed_data = df.copy()
        
        # Creating a list of dummy column named as columns to omit
        columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Education']
        
        # Separate the dataframe into dummy and non-dummy dataframes
        dummy_df = df.loc[:, df.columns.isin(columns_to_omit)] 
        non_dummy_df = df.loc[:, ~df.columns.isin(columns_to_omit)]
        
        # transform the Non-dummy inputs and convert it into a DataFrame
        scaled_non_dummy_df = pd.DataFrame(self.scaler.transform(non_dummy_df), columns = non_dummy_df.columns)
        
        # Join dummy and scaled non-dummy dataframes
        df = pd.concat([dummy_df, scaled_non_dummy_df], axis=1)[df.columns]
        
        # store the transformed data into new variable for later use
        self.data = df.copy() # we use this variable in next functions
        
        return self.data
        
    # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if (self.data is not None):  
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred

    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

    # Above two functions "predicted_probability()" and "predicted_output_category()" are sum into belowed function:
    # It predict the outputs and the probabilities and 
    # add columns with these values at the end of the new data
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Predicted probability'] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data['Predicted outputs'] = self.reg.predict(self.data)
            return self.preprocessed_data

