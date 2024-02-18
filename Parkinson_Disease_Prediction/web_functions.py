"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score



@st.cache_data()
def load_data():
    """This function returns the preprocessed data"""

    df = pd.read_csv('Parkinson.csv')

    # Rename the column names in the DataFrame.
    df.rename(columns = {"MDVP:Fo(Hz)": "AVFF",}, inplace = True)
    df.rename(columns = {"MDVP:Fhi(Hz)": "MAVFF",}, inplace = True)
    df.rename(columns = {"MDVP:Flo(Hz)": "MIVFF",}, inplace = True)
    

    # Perform feature and target split
    X = df[["AVFF", "MAVFF", "MIVFF","Jitter:DDP","MDVP:Jitter(%)","MDVP:RAP","MDVP:APQ","MDVP:PPQ","MDVP:Shimmer","Shimmer:DDA","Shimmer:APQ3","Shimmer:APQ5","NHR","HNR","RPDE","DFA","D2","PPE"]]
    y = df['status']

    return df, X, y

@st.cache_data()
def train_model(X, y):
    
    # Create the model
    params = {
    'objective': 'binary:logistic',  # For binary classification
    'max_depth': 4, 
    'learning_rate': 0.1,  # Learning rate
    'subsample': 0.8,  # Subsample ratio of the training instance
    'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
    'random_state': 42
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(**params)
    
    # Fit the data on model and calculate score
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    # Return the values
    return model, score


def predict(X, y, features):
    
    # Get model and model score
    model, score = train_model(X, y)
    
    # Predict the value
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score
