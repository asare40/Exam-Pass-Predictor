# -*- coding: utf-8 -*-
"""
Created on Tue May  6 00:00:58 2025

@author: kings
"""

import pandas as pd
import numpy as np

# Create sample data
data = {
    'Student_ID': range(1, 101),
    'Age': np.random.randint(16, 22, 100),
    'Gender': np.random.choice(['Male', 'Female'], 100),
    'School_Type': np.random.choice(['Public', 'Private', 'Federal'], 100),
    'Socioeconomic_Status': np.random.choice(['Low', 'Middle', 'High'], 100),
    'Study_Hours': np.random.randint(1, 15, 100),
    'Attendance_Rate': np.random.uniform(50, 100, 100),
    'Previous_Score': np.random.randint(40, 95, 100),
    'Extracurricular': np.random.choice(['Yes', 'No'], 100),
    'Sleep_Hours': np.random.uniform(4, 9, 100),
    'JAMB_Score': np.random.randint(120, 350, 100)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Add Pass_Status (pass if JAMB_Score >= 200)
df['Pass_Status'] = (df['JAMB_Score'] >= 200).astype(int)

# Save to CSV
df.to_csv('enhanced_jamb_results.csv', index=False)
print("Sample dataset created: enhanced_jamb_results.csv")
