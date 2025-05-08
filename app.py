from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
from io import BytesIO
import base64
import os
import joblib

# Initialize Flask application
app = Flask(__name__)

# Define dataset and model paths
DATASET_PATH = 'enhanced_jamb_results.csv'
MODEL_PATH = 'jamb_prediction_model.pkl'

# Global variables to track if model and dataset are loaded
model_loaded = False
dataset_loaded = False
model = None

def get_dataframe():
    """Load the dataset from CSV file"""
    global dataset_loaded
    try:
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            dataset_loaded = True
            return df
        else:
            print(f"Dataset not found at {DATASET_PATH}")
            dataset_loaded = False
            return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        dataset_loaded = False
        return None

def load_model():
    """Load the trained model"""
    global model, model_loaded
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            model_loaded = True
            return True
        else:
            print(f"Model not found at {MODEL_PATH}")
            model_loaded = False
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False
        return False

@app.route('/')
def home():
    """Home page route"""
    df = get_dataframe()
    numerical_features = []
    categorical_features = []
    if df is not None:
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Remove target variables from features lists
        for col in ['Student_ID', 'JAMB_Score', 'Pass_Status']:
            if col in numerical_features:
                numerical_features.remove(col)
            if col in categorical_features:
                categorical_features.remove(col)
    
    return render_template('index.html', 
                          model_loaded=model_loaded, 
                          dataset_loaded=dataset_loaded,
                          numerical_features=numerical_features,
                          categorical_features=categorical_features)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction form submission"""
    if not model_loaded:
        return render_template('error.html', message="Model not loaded. Please run setup first.")
    
    try:
        # Get all form data
        student_data = {}
        for key in request.form:
            value = request.form[key]
            # Convert numeric values
            if key in ['Age', 'Study_Hours', 'Attendance_Rate', 'Previous_Score', 'Sleep_Hours']:
                student_data[key] = float(value)
            else:
                student_data[key] = value
        
        # Convert to DataFrame for prediction
        student_df = pd.DataFrame([student_data])
        
        # Make prediction
        probability = model.predict_proba(student_df)[0][1] * 100  # Probability of passing
        prediction = "Pass" if probability >= 50 else "Fail"
        
        return render_template('result.html', 
                              prediction=prediction,
                              probability=round(probability, 1),
                              student_data=student_data)
    
    except Exception as e:
        return render_template('error.html', message=f"Error making prediction: {str(e)}")

@app.route('/feature-analysis')
def feature_analysis_page():
    """Feature analysis selection page"""
    df = get_dataframe()
    if df is None:
        return render_template('error.html', message="Dataset not loaded. Please run setup first.")
    
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target variables from features lists
    for col in ['Student_ID', 'JAMB_Score', 'Pass_Status']:
        if col in numerical_features:
            numerical_features.remove(col)
        if col in categorical_features:
            categorical_features.remove(col)
    
    return render_template('feature_analysis_select.html', 
                          numerical_features=numerical_features,
                          categorical_features=categorical_features)

@app.route('/feature-analysis', methods=['POST'])
def feature_analysis_result():
    """Process feature analysis form submission"""
    feature_name = request.form.get('feature')
    if not feature_name:
        return render_template('error.html', message="No feature selected.")
    
    return analyze_feature(feature_name)

def analyze_feature(feature_name):
    """
    Analyze a single feature's impact on JAMB score and generate visualizations with detailed explanations.
    """
    try:
        # Get the dataset
        df = get_dataframe()
        if df is None:
            return render_template('error.html', message="Dataset not loaded. Please run setup first.")
        
        # Check if feature exists in dataset
        if feature_name not in df.columns:
            return render_template('error.html', message=f"Feature '{feature_name}' not found in dataset.")
        
        # Create a pass status column (1 for pass, 0 for fail)
        pass_threshold = 180  # Define threshold for passing
        pass_status = df['JAMB_Score'] >= pass_threshold
        
        # Ensure pass_status is numeric (1 for True, 0 for False)
        df['pass_numeric'] = pass_status.astype(int)
        
        # Calculate correlation for numerical features
        correlation_message = None
        if feature_name in df.select_dtypes(include=['int64', 'float64']).columns:
            correlation = df[feature_name].corr(df['JAMB_Score'])
            correlation_strength = ""
            if abs(correlation) < 0.2:
                correlation_strength = "weak"
            elif abs(correlation) < 0.5:
                correlation_strength = "moderate"
            else:
                correlation_strength = "strong"
                
            correlation_direction = "positive" if correlation > 0 else "negative"
            
            correlation_message = f"The correlation between {feature_name} and JAMB Score is {correlation:.3f}, indicating a {correlation_strength} {correlation_direction} relationship."
            if correlation > 0:
                correlation_message += f" This suggests that as {feature_name} increases, JAMB scores tend to increase as well."
            else:
                correlation_message += f" This suggests that as {feature_name} increases, JAMB scores tend to decrease."
            
            # Create bins for numerical features
            try:
                num_bins = min(5, len(df[feature_name].unique()))  # Avoid empty bins
                df['temp_bins'] = pd.qcut(df[feature_name], num_bins, duplicates='drop')
                
                # Calculate pass rate for each bin - use numeric pass status
                pass_rates = df.groupby('temp_bins')['pass_numeric'].mean() * 100
                
                # Calculate mean JAMB score for each bin
                mean_scores = df.groupby('temp_bins')['JAMB_Score'].mean()
                
                # Create the plot with both pass rates and mean scores
                fig = plt.figure(figsize=(10, 6))
                ax1 = fig.add_subplot(111)
                
                # Plot pass rates as bars
                pass_rates.plot(kind='bar', color='skyblue', ax=ax1)
                ax1.set_title(f'Pass Rate & Mean JAMB Score by {feature_name}')
                ax1.set_xlabel(feature_name)
                ax1.set_ylabel('Pass Rate (%)', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                
                # Create second y-axis for mean scores
                ax2 = ax1.twinx()
                mean_scores.plot(kind='line', marker='o', color='red', ax=ax2)
                ax2.set_ylabel('Mean JAMB Score', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Format the pass rates into a readable message
                bins_labels = [f"{b.left:.1f}-{b.right:.1f}" for b in pass_rates.index]
                pass_rate_message = f"Analyzing pass rates by {feature_name} reveals important patterns:\n"
                
                for i, (bin_label, rate, score) in enumerate(zip(bins_labels, pass_rates, mean_scores)):
                    pass_rate_message += f"- {bin_label}: {rate:.1f}% pass rate (mean JAMB score: {score:.1f})\n"
                
                # Calculate the range with the highest pass rate
                best_bin_index = pass_rates.argmax()
                best_bin = pass_rates.index[best_bin_index]
                best_range = f"{best_bin.left:.1f}-{best_bin.right:.1f}"
                
                # Generate detailed recommendation based on the trend
                if correlation > 0.2:
                    recommendation = f"Higher {feature_name} is strongly associated with better JAMB performance. Students with {feature_name} in the range of {best_range} showed the highest pass rate of {pass_rates.max():.1f}%. Consider strategies to increase this factor towards this optimal range."
                elif correlation < -0.2:
                    recommendation = f"Lower {feature_name} is associated with better JAMB performance. Students with {feature_name} in the range of {best_range} showed the highest pass rate of {pass_rates.max():.1f}%. Consider strategies to optimize this factor within this effective range."
                else:
                    recommendation = f"The relationship between {feature_name} and JAMB performance is not very strong, but students with {feature_name} in the range of {best_range} showed the highest pass rate of {pass_rates.max():.1f}%. Other factors may have more significant impacts on your performance."
                
            except Exception as e:
                print(f"Error in binning: {e}")
                # Alternative if binning fails - create a scatter plot
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(df[feature_name], df['JAMB_Score'], alpha=0.5, color='blue')
                
                # Add trend line
                z = np.polyfit(df[feature_name], df['JAMB_Score'], 1)
                p = np.poly1d(z)
                plt.plot(df[feature_name], p(df[feature_name]), "r--", alpha=0.8)
                
                plt.title(f'{feature_name} vs JAMB Score with Trend Line')
                plt.xlabel(feature_name)
                plt.ylabel('JAMB Score')
                plt.axhline(y=pass_threshold, color='green', linestyle='-', label='Pass Threshold (180)')
                plt.legend()
                plt.tight_layout()
                
                pass_rate_message = f"Unable to calculate pass rates in bins for {feature_name}, so a scatter plot is shown instead. Each point represents a student's {feature_name} value and corresponding JAMB score. The red dashed line shows the overall trend."
                recommendation = f"Based on the scatter plot, there appears to be a {'positive' if correlation > 0 else 'negative'} relationship between {feature_name} and JAMB Score. Consider the general trend when planning your study strategy."
                
        # Handle categorical features
        else:
            # For categorical features, calculate pass rate for each category
            category_pass_rates = df.groupby(feature_name)['pass_numeric'].mean() * 100
            
            # Calculate mean JAMB scores for each category
            category_mean_scores = df.groupby(feature_name)['JAMB_Score'].mean()
            
            # Calculate count of students in each category
            category_counts = df.groupby(feature_name).size()
            
            # Create the plot
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(111)
            
            # Plot pass rates as bars
            category_pass_rates.plot(kind='bar', color='lightgreen', ax=ax1)
            ax1.set_title(f'Pass Rate & Mean JAMB Score by {feature_name}')
            ax1.set_xlabel(feature_name)
            ax1.set_ylabel('Pass Rate (%)', color='green')
            ax1.tick_params(axis='y', labelcolor='green')
            
            # Create second y-axis for mean scores
            ax2 = ax1.twinx()
            category_mean_scores.plot(kind='line', marker='o', color='purple', ax=ax2)
            ax2.set_ylabel('Mean JAMB Score', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Format pass rates into a readable message with mean scores and sample sizes
            pass_rate_message = f"Analyzing {feature_name} categories reveals the following patterns:\n"
            for category, rate in category_pass_rates.items():
                mean_score = category_mean_scores[category]
                count = category_counts[category]
                pass_rate_message += f"- {category}: {rate:.1f}% pass rate (mean score: {mean_score:.1f}, sample size: {count})\n"
            
            # Find the category with the highest pass rate
            best_category = category_pass_rates.idxmax()
            worst_category = category_pass_rates.idxmin()
            
            # Calculate the difference between best and worst
            difference = category_pass_rates[best_category] - category_pass_rates[worst_category]
            
            if difference > 10:  # Significant difference
                recommendation = f"The '{best_category}' category in {feature_name} shows the highest pass rate at {category_pass_rates[best_category]:.1f}%, which is {difference:.1f}% higher than the lowest category ('{worst_category}'). This suggests that {feature_name} may have a substantial influence on JAMB performance."
            else:  # Small difference
                recommendation = f"While the '{best_category}' category in {feature_name} shows the highest pass rate at {category_pass_rates[best_category]:.1f}%, the difference between categories is relatively small ({difference:.1f}%). This suggests that other factors might have a stronger influence on JAMB performance than {feature_name}."
            
        # Save the plot to a bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()  # Close the plot to free memory
        
        # Remove temporary bins column
        if 'temp_bins' in df.columns:
            df.drop('temp_bins', axis=1, inplace=True)
        
        # Remove temporary pass_numeric column
        if 'pass_numeric' in df.columns:
            df.drop('pass_numeric', axis=1, inplace=True)
            
        return render_template('feature_analysis.html', 
                              feature=feature_name,
                              plot_url=plot_data,
                              correlation_message=correlation_message,
                              pass_rate_message=pass_rate_message,
                              recommendation=recommendation)
    
    except Exception as e:
        error_message = f"Error analyzing feature: {str(e)}"
        return render_template('error.html', message=error_message)

@app.route('/model-info')
def model_info():
    """Display model information page with feature importance"""
    if not model_loaded:
        return render_template('error.html', message="Model not loaded. Please run setup first.")
    
    try:
        # Get feature importance from the model
        importance = model.named_steps['classifier'].feature_importances_
        
        # Get feature names
        feature_names = []
        
        # Extract numerical feature names
        for name in model.named_steps['preprocessor'].transformers_[0][2]:
            feature_names.append(name)
        
        # Extract categorical feature names with their encoded values
        categorical_features = model.named_steps['preprocessor'].transformers_[1][2]
        ohe = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
        for i, category in enumerate(categorical_features):
            encoded_features = ohe.categories_[i]
            for value in encoded_features:
                feature_names.append(f"{category}_{value}")
        
        # Create list of features and their importance
        feature_importances = [(name, imp) for name, imp in zip(feature_names, importance)]
        feature_importances.sort(key=lambda x: x[1], reverse=True)
        
        # Create importance plot
        plt.figure(figsize=(10, 6))
        names = [x[0] for x in feature_importances[:10]]  # Top 10 features
        values = [x[1] for x in feature_importances[:10]]
        
        plt.barh(range(len(names)), values, align='center')
        plt.yticks(range(len(names)), names)
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        
        # Save the plot to a bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return render_template('model_info.html', 
                              importance_plot=plot_data,
                              feature_importances=feature_importances)
    
    except Exception as e:
        return render_template('error.html', message=f"Error getting model info: {str(e)}")

@app.route('/setup')
def setup():
    """Run setup operations to create sample data and model"""
    try:
        # Create sample dataset if it doesn't exist
        dataset_message = ""
        if os.path.exists(DATASET_PATH):
            dataset_message = "Dataset already exists. Using existing dataset."
            dataset_loaded = True
        else:
            try:
                import create_sample_data
                dataset_message = "Sample dataset created successfully."
                dataset_loaded = True
            except Exception as e:
                dataset_message = f"Error creating sample dataset: {str(e)}"
                dataset_loaded = False
        
        # Create model if it doesn't exist
        model_message = ""
        if os.path.exists(MODEL_PATH):
            model_message = "Model already exists. Using existing model."
            model_loaded = True
        else:
            try:
                import create_sample_model
                model_message = "Sample model created successfully."
                model_loaded = True
            except Exception as e:
                model_message = f"Error creating sample model: {str(e)}"
                model_loaded = False
        
        # Load model and dataset
        load_model()
        get_dataframe()
        
        return render_template('setup.html', 
                              dataset_message=dataset_message,
                              model_message=model_message)
    
    except Exception as e:
        return render_template('error.html', message=f"Error during setup: {str(e)}")

@app.route('/about')
def about():
    """Display about page"""
    return render_template('about.html')

if __name__ == '__main__':
    # Try to load model and dataset at startup
    load_model()
    get_dataframe()
    app.run(debug=True)
