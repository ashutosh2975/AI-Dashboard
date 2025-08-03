from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load('chart_model.pkl')

# Folder to save uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_features(df):
    num_columns = len(df.columns)
    num_rows = len(df)
    num_numerical = len(df.select_dtypes(include='number').columns)
    num_categorical = len(df.select_dtypes(include='object').columns)
    has_datetime = 1 if any(df.dtypes == 'datetime64[ns]') else 0
    has_geo = 1 if any(col.lower() in ['latitude', 'longitude', 'lat', 'lon', 'geo', 'location'] for col in df.columns) else 0
    avg_unique_values = df.nunique().mean()
    
    # Compute correlation score (if enough numerical columns)
    numerical_df = df.select_dtypes(include='number')
    correlation_score = 0
    if numerical_df.shape[1] >= 2:
        correlation_score = abs(numerical_df.corr().values[np.triu_indices(numerical_df.shape[1], k=1)]).mean()
    
    features = pd.DataFrame([[
        num_columns, num_rows, num_numerical, num_categorical,
        has_datetime, has_geo, avg_unique_values, correlation_score
    ]], columns=[
        'num_columns', 'num_rows', 'num_numerical', 'num_categorical',
        'has_datetime', 'has_geo', 'avg_unique_values', 'correlation_score'
    ])
    
    return features

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
    except Exception as e:
        return jsonify({'error': f'File read error: {str(e)}'}), 500

    # Convert object-type datetime columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    # Extract features and predict
    features = extract_features(df)
    prediction = model.predict(features)[0]

    # Chart type labels
    chart_labels = ['bar', 'pie', 'line', 'scatter', 'table', 'map']
    recommended_charts = [chart_labels[i] for i, val in enumerate(prediction) if val == 1]

    return jsonify({
        'recommended_charts': recommended_charts,
        'message': 'Charts predicted successfully!'
    })

@app.route('/analyze', methods=['POST'])
def analyze_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file)

        column_summaries = []

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                summary = {
                    'name': col,
                    'type': 'numeric',
                    'data': df[col].dropna().tolist()
                }
            elif pd.api.types.is_string_dtype(df[col]):
                value_counts = df[col].value_counts().to_dict()
                summary = {
                    'name': col,
                    'type': 'categorical',
                    'data': value_counts
                }
            else:
                summary = {
                    'name': col,
                    'type': 'unknown',
                    'data': {}
                }

            column_summaries.append(summary)

        return jsonify({'columns': column_summaries})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
