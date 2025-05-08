#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for JAMB Score Prediction System
This script creates the proper directory structure and moves files to the correct locations.
"""

import os
import shutil

def setup_project():
    """
    Create the proper directory structure for the Flask application.
    """
    print("Setting up JAMB Score Prediction System project...")
    
    # Create directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    
    # List of HTML files to move to templates directory
    template_files = [
        'about.html',
        'error.html',
        'feature_analysis.html',
        'feature_analysis_select.html',
        'index.html',
        'model_info.html',
        'result.html',
        'setup.html'
    ]
    
    # Move HTML files to templates directory
    for template_file in template_files:
        if os.path.exists(template_file):
            print(f"Moving {template_file} to templates directory...")
            shutil.move(template_file, os.path.join('templates', template_file))
        else:
            print(f"Warning: {template_file} not found in current directory.")
    
    # Create CSS file if it doesn't exist
    css_path = os.path.join('static', 'css', 'style.css')
    if not os.path.exists(css_path):
        print("Creating style.css...")
        with open(css_path, 'w') as f:
            f.write("""
/* Main Styles */
body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f5f5f5;
    color: #333;
}

h1, h2, h3 {
    color: #2c3e50;
}

nav {
    background-color: #2c3e50;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
}

nav a {
    color: white;
    text-decoration: none;
    padding: 10px 15px;
    margin-right: 5px;
    border-radius: 3px;
    transition: background-color 0.3s;
}

nav a:hover {
    background-color: #34495e;
}

.container, .prediction-section, .feature-section, .about-section, .model-info, .result-container, .student-data {
    background-color: white;
    padding: 20px;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* Form Styles */
form {
    display: grid;
    grid-gap: 15px;
}

form h3 {
    margin-top: 20px;
    border-bottom: 1px solid #eee;
    padding-bottom: 5px;
}

label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
}

input, select {
    width: 100%;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    box-sizing: border-box;
}

button {
    background-color: #2c3e50;
    color: white;
    padding: 10px 15px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s;
}

button:hover {
    background-color: #34495e;
}

/* Result Styles */
.pass {
    color: green;
}

.fail {
    color: red;
}

.probability-bar {
    background-color: #eee;
    border-radius: 5px;
    height: 25px;
    margin: 20px 0;
    overflow: hidden;
}

.probability-fill {
    height: 100%;
    text-align: center;
    color: white;
    line-height: 25px;
    transition: width 1s ease-in-out;
}

/* Table Styles */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

th, td {
    text-align: left;
    padding: 10px;
    border-bottom: 1px solid #ddd;
}

th {
    background-color: #f2f2f2;
}

/* Alert Styles */
.alert {
    background-color: #f8d7da;
    color: #721c24;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
}

.alert a {
    color: #721c24;
    font-weight: bold;
}

/* Responsive Design */
@media (max-width: 768px) {
    form {
        grid-template-columns: 1fr;
    }
    
    nav a {
        display: block;
        margin-bottom: 5px;
    }
}
""")
    
    # Check if important files exist
    required_files = [
        'app.py',
        'create_sample_data.py',
        'create_sample_model.py'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Warning: {file} not found in current directory.")
    
    print("\nSetup complete!")
    print("\nTo run the application:")
    print("1. Make sure you have all required packages installed:")
    print("   pip install -r requirements.txt")
    print("2. Run the Flask app:")
    print("   python app.py")
    print("3. Visit http://127.0.0.1:5000/ in your browser")
    print("4. On first run, visit http://127.0.0.1:5000/setup to create sample data and train the model")

if __name__ == "__main__":
    setup_project()
