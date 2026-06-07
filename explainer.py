import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

# Force matplotlib to run headlessly (fixes Flask threading crashes)
matplotlib.use('Agg') 

def generate_shap_chart(disease, patient_data_dict):
    print(f"Generating SHAP Explanation for {disease}...")
    
    # 1. Load the pre-trained pipeline and feature names
    pipeline = joblib.load(f'{disease}_pipeline.pkl')
    features = joblib.load(f'{disease}_features.pkl')
    
    # 2. Extract the model and scaler from the pipeline
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['model']
    
    # 3. Format the patient data to match the exact training shape
    patient_df = pd.DataFrame([patient_data_dict])
    
    # Auto-fill any missing columns with 0
    for col in features:
        if col not in patient_df.columns:
            patient_df[col] = 0
            
    # Ensure column order is perfect
    patient_df = patient_df[features]
    
    # 4. Scale the math
    scaled_input = scaler.transform(patient_df)
    
    # 5. Calculate Game Theory Values (SHAP)
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(scaled_input)
    
    # 🚀 THE INDESTRUCTIBLE SHAPE FIX
    
    # Part A: Format the Risk Values
    if isinstance(shap_values_raw, list):
        risk_values = np.array(shap_values_raw[1]).flatten()
    else:
        # Fallback for weird 3D array shapes
        np_shap = np.array(shap_values_raw)
        if len(np_shap.shape) == 3:
            risk_values = np_shap[0, :, 1].flatten()
        else:
            risk_values = np_shap.flatten()
            
    # Part B: Crush the Base Value
    # np.ravel destroys nested arrays (e.g., [[0.55, 0.45]] becomes [0.55, 0.45])
    flat_base = np.ravel(explainer.expected_value)
    
    # The positive class probability is always the last number in the crushed array
    final_base_value = float(flat_base[-1])
    
    # 6. Draw the Professional Chart
    plt.figure(figsize=(10, 6))
    
    exp = shap.Explanation(
        values=risk_values, 
        base_values=final_base_value, 
        data=patient_df.iloc[0].values,
        feature_names=features
    )
    
    shap.plots.waterfall(exp, show=False, max_display=10)
    
    # 7. Save it for the frontend
    os.makedirs('static/shap_charts', exist_ok=True)
    chart_path = f'static/shap_charts/shap_{disease}.png'
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_path