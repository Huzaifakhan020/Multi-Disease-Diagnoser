import pandas as pd
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

print("Booting Enterprise Modular Training Pipeline...")

with open('manifest.json') as f:
    config = json.load(f)

for disease, target in config['diseases'].items():
    try:
        # 1. Load the dataset
        df = pd.read_csv(f'{disease}.csv')
        
        # 2. Bulletproof Header Cleanup
        df.columns = df.columns.str.strip()
        if 'Target' in df.columns and target == 'target':
            df.rename(columns={'Target': 'target'}, inplace=True)
            
        # 3. Clean Missing Data (Drop empty rows)
        df = df.dropna()
            
        # 4. Split features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        # 5. AUTO-ENCODE: Convert text columns (like "normal") into numbers (0s and 1s)
        X = pd.get_dummies(X, drop_first=True)
        
        # Encode target variable if it is text instead of numbers
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        
        # 6. Build and train the pipeline
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('model', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
        ])
        pipeline.fit(X, y)
        
        # 7. Export the trained pipeline and feature names
        joblib.dump(pipeline, f'{disease}_pipeline.pkl')
        joblib.dump(list(X.columns), f'{disease}_features.pkl')
        
        print(f"✅ Successfully compiled {disease.upper()} pipeline.")
        
    except FileNotFoundError:
        print(f"⚠️ Warning: {disease}.csv not found in folder. Skipping...")
    except Exception as e:
        print(f"❌ Error compiling {disease}: {e}")

print("Training cycle complete!")