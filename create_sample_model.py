# -*- coding: utf-8 -*-
"""
Created on Tue May  6 00:03:53 2025

@author: kings
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('enhanced_jamb_results.csv')

# Separate features and target
X = df.drop(['Student_ID', 'JAMB_Score', 'Pass_Status'], axis=1)
y = df['Pass_Status']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create and train the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=1000, random_state=42))
])

model.fit(X, y)

# Save the model
joblib.dump(model, 'jamb_prediction_model.pkl')
print("Sample model created: jamb_prediction_model.pkl")
