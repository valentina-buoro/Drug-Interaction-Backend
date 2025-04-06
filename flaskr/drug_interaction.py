from flask import Blueprint, Flask, jsonify,request
import subprocess
import pandas as pd
import joblib
import os
import time



bp = Blueprint('drug_interaction', __name__, url_prefix='/drug')

# Load trained XGBoost model
model = joblib.load("flaskr/xgboost_model.pkl")

def featurize_smiles(smiles1, smiles2):
    """Featurize two SMILES strings using Ersilia binary Morgan fingerprint."""
    
    def run_featurization(smiles, label,model_id):
        print(f"Fetching model {model_id}...")
        fetch_result = subprocess.run(["ersilia", "fetch", model_id], capture_output=True, text=True)
        if fetch_result.returncode != 0:
            print(f" Error fetching model: {fetch_result.stderr}")
            return None  
        print("Model fetched successfully")

        server_process = subprocess.run(["ersilia", "serve", model_id])
        time.sleep(15)
        print("Ersilia model server is running")

        temp_input = f"{label}.csv"
        temp_output = f"{label}_feat.csv"

        pd.DataFrame({"SMILES": [smiles]}).to_csv(temp_input, index=False)

        result = subprocess.run(["ersilia", "run", "-i", temp_input, "-o", temp_output],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                                   
        print(f"[{label}] stdout:\n{result.stdout}")
        print(f"[{label}] stderr:\n{result.stderr}")

        if result.returncode != 0:
            print(f"[{label}] Ersilia run failed:\n{result.stderr}")
            return None, result.stderr

    # Check if output file actually exists
        if not os.path.exists(temp_output):
            print(f"[{label}] Output file not found: {temp_output}")
            return None, f"File {temp_output} not created"

        df_feat = pd.read_csv(temp_output)
        df_feat.columns = [f"{label}_{col}" for col in df_feat.columns]
        df_feat = df_feat.drop(columns=[f'{label}_key', f'{label}_input'])
        df_feat.columns = [f"{label}_fps-{i}" for i in range(df_feat.shape[1])]
   
        return df_feat, None
        

    feat1, err1 = run_featurization(smiles1, "Drug1", "eos4wt0")
    feat2, err2 = run_featurization(smiles2, "Drug2","eos4wt0")

    if err1:
        return None, f"Error in Drug1 featurization: {err1}"
    if err2:
        return None, f"Error in Drug2 featurization: {err2}"

    combined_feat = pd.concat([feat1, feat2], axis=1)
   
    return combined_feat, None




@bp.route('/predict', methods=["POST"])
def predict():
    data = request.json
    smiles1 = data.get("smiles1")
    smiles2 = data.get("smiles2")

    
    if not smiles1 or not smiles2:
        return jsonify({"error": "No SMILES string provided"}), 400
    
    features, error = featurize_smiles(smiles1, smiles2)
    if error:
        return jsonify({"error": f"Featurization failed: {error}"}), 500
    
    prediction = model.predict(features)
    
    return jsonify({"smiles1": smiles1, "smiles2": smiles2, "prediction": (prediction + 1).tolist()})





