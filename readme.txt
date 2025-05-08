# JAMB Score Prediction System

A machine learning web application to predict students' performance in the Joint Admissions and Matriculation Board (JAMB) examination.

## Features

- **Prediction System**: Input student details to predict their likelihood of passing JAMB
- **Feature Analysis**: Analyze how different factors impact JAMB performance
- **Data Visualization**: Visual representations of feature importance and relationships
- **API Support**: REST API endpoint for integration with other applications

## Installation

1. Clone the repository or download the files
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Setup and Running

1. Make sure all files are in the correct structure (see below)
2. Run the Flask app:
   ```
   python app.py
   ```
3. Access the app in your browser at: http://127.0.0.1:5000/
4. First-time setup: Visit http://127.0.0.1:5000/setup to create sample data and model

## First-time Usage

1. Run the setup route: http://127.0.0.1:5000/setup
2. This will create the sample dataset and train a machine learning model
3. Return to the main page to start using the application

## File Structure

Make sure your project has the following structure:

```
jamb_prediction_system/
├── app.py                     # Main Flask application
├── create_sample_data.py      # Script to create sample dataset
├── create_sample_model.py     # Script to train the model
├── enhanced_jamb_results.csv  # Dataset (will be created if missing)
├── jamb_prediction_model.pkl  # Model file (will be created if missing)
├── requirements.txt           # Python dependencies
├── troubleshoot.py            # Troubleshooting script
├── static/
│   └── css/
│       └── style.css          # CSS styling
└── templates/
    ├── about.html             # About page
    ├── error.html             # Error page
    ├── feature_analysis.html  # Feature analysis results
    ├── feature_analysis_select.html  # Feature selection
    ├── index.html             # Home page
    ├── model_info.html        # Model information
    ├── result.html            # Prediction results
    └── setup.html             # Setup page
```

## Troubleshooting

If you encounter any issues:

1. Run the troubleshooting script:
   ```
   python troubleshoot.py
   ```
2. Make sure all templates are in the templates directory
3. Ensure the static/css directory contains style.css
4. Verify the model and dataset files exist, or run /setup route
5. Check the console for any error messages

## API Usage

The system provides a REST API endpoint for programmatic access:

```
POST /api/predict
Content-Type: application/json

{
  "Age": 18,
  "Gender": "Male",
  "School_Type": "Public",
  "Socioeconomic_Status": "Middle",
  "Study_Hours": 8,
  "Attendance_Rate": 85,
  "Previous_Score": 75,
  "Extracurricular": "Yes",
  "Sleep_Hours": 7
}
```

Example response:
```json
{
  "prediction": "Pass",
  "probability": 78.5,
  "student_data": {
    "Age": 18,
    "Gender": "Male",
    "School_Type": "Public",
    "Socioeconomic_Status": "Middle",
    "Study_Hours": 8,
    "Attendance_Rate": 85,
    "Previous_Score": 75,
    "Extracurricular": "Yes",
    "Sleep_Hours": 7
  }
}
```

## About JAMB

The Joint Admissions and Matriculation Board (JAMB) is a Nigerian entrance examination board for tertiary-level institutions. The board conducts entrance examinations for prospective undergraduates into Nigerian universities.

## License

This project is open source and available for educational purposes.
