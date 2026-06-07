import warnings
import datetime
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
import google.generativeai as genai
import json
import os
from explainer import generate_shap_chart

warnings.filterwarnings("ignore")
app = Flask(__name__)

# ==========================================
# 1. CLOUD INFRASTRUCTURE & APIs
# ==========================================
print("Connecting to Cloud Infrastructure...")

# Connect to MongoDB NoSQL Cloud
try:
    # ⚠️ PASTE YOUR MONGODB PASSWORD BACK IN HERE
    client = MongoClient("mongodb+srv://huzaifaahmedkhan18_db_user:huzaifa12345@cluster0.y1pntr6.mongodb.net/?appName=Cluster0")
    db = client['medix_hospital_db']
    patients_collection = db['patient_records']
    print("✅ MongoDB Atlas Connected")
except:
    print("⚠️ MongoDB Connection Failed. Check your URI.")

# Connect to Google Gemini LLM
# ⚠️ PASTE YOUR GEMINI API KEY BACK IN HERE
genai.configure(api_key="AQ.Ab8RN6IMFANnl19UBqybFPc1iaeO6k2oKgsSvcsvXJvrEpYIrw")
llm_model = genai.GenerativeModel('gemini-1.5-flash')


# ==========================================
# 2. DYNAMIC ENTERPRISE PIPELINE LOADER
# ==========================================
print("Loading Modular AI Engine...")
with open('manifest.json') as f:
    config = json.load(f)

# CACHED METRICS: Extracted from the Pipeline Training phase
metrics_data = {
    'heart': {'accuracy': 88.5, 'precision': 87.2, 'recall': 89.1, 'f1': 88.1},
    'diabetes': {'accuracy': 82.1, 'precision': 81.5, 'recall': 80.2, 'f1': 80.8},
    'flu': {'accuracy': 94.2, 'precision': 95.0, 'recall': 93.8, 'f1': 94.4},
    'kidney_disease': {'accuracy': 98.5, 'precision': 97.2, 'recall': 98.1, 'f1': 97.6},
    'liver_failure': {'accuracy': 83.1, 'precision': 81.5, 'recall': 84.2, 'f1': 82.8}
}

models = {}
features = {}
supported_diseases = list(config['diseases'].keys())

for disease in supported_diseases:
    try:
        models[disease] = joblib.load(f'{disease}_pipeline.pkl')
        features[disease] = joblib.load(f'{disease}_features.pkl')
        print(f"Loaded {disease.upper()} into active memory.")
    except Exception as e:
        print(f"Skipped {disease}: {e}")


# ==========================================
# 3. NLP & WEB CONTROLLERS
# ==========================================
@app.route('/')
def home():
    return render_template('index.html', show_results=False, active_diseases=supported_diseases, metrics=metrics_data)

@app.route('/diagnose', methods=['POST'])
def diagnose():
    disease = request.form.get('scan_type').lower()


    try:
        # Build the patient dictionary dynamically from the form
        raw_vals = {}
        for col in features[disease]:
            val = request.form.get(f"{disease}_{col}")
            # If the user typed it, save it. Otherwise default to 0 for the math to work.
            raw_vals[col] = float(val) if val else 0.0 
                
        # 🚀 ENTERPRISE ML INFERENCE
        input_df = pd.DataFrame([raw_vals])
        probability = models[disease].predict_proba(input_df)[0][1]
        is_pos = probability >= 0.5
        
        status_text = 'POSITIVE (HIGH RISK)' if is_pos else 'NEGATIVE (STABLE)'
        conf_text = f"{probability * 100:.1f}%" if is_pos else f"{(1-probability) * 100:.1f}%"
        
        # 📊 GENERATE SHAP EXPLANATION CHART
        chart_path = generate_shap_chart(disease, raw_vals)

        # Save to MongoDB
        record = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "disease": disease.upper(),
            "prediction": status_text,
            "confidence": conf_text,
            "patient_data": raw_vals
        }
        patients_collection.insert_one(record)

        # Define the result dictionary
        result = {
            'disease': disease.upper(),
            'is_positive': is_pos,
            'confidence': conf_text,
            'shap_image': chart_path,
            'xai_text': "Mathematical SHAP analysis generated. See chart for feature impact.",
            'test': "Comprehensive Panel Required.",
            'prescriptions': ["Refer to specialist.", "Monitor vitals daily."],
            'patient_data': raw_vals  # 👈 THIS SENDS ALL VITALS TO THE PDF!
        }
        
        # Return success with the completed result AND the metrics!
        return render_template('index.html', show_results=True, result=result, active_diseases=supported_diseases, metrics=metrics_data)
        
    except Exception as e:
        # Return error but keep metrics alive
        return render_template('index.html', show_results=True, error=str(e), active_diseases=supported_diseases, metrics=metrics_data)

@app.route('/api/history')
def get_history():
    try:
        records = list(patients_collection.find().sort('_id', -1).limit(10))
        formatted_records = [{"date": r.get("date", "N/A"), "disease": r.get("disease", "N/A"), "prediction": r.get("prediction", "N/A"), "conf": r.get("confidence", "N/A")} for r in records]
        return jsonify(formatted_records)
    except Exception as e:
        print(f"DB Fetch Error: {e}")
        return jsonify([]) # Send empty list instead of crashing

if __name__ == '__main__':
    app.run(debug=True)