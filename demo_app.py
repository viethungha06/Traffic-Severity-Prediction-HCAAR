import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load trained model and column list
def load_model():
    model_path = 'output/rf_model_m1.pkl'
    cols_path  = 'output/model_columns_m1.pkl'

    if not os.path.exists(model_path) or not os.path.exists(cols_path):
        print("[!] Model not found. Please run ml_prediction.py first.")
        return None, None

    model         = joblib.load(model_path)
    model_columns = joblib.load(cols_path)
    return model, model_columns

# Step 2: Collect user input and run prediction
def predict_m1_severity(model, model_columns):
    print("\n" + "="*70)
    print(" 🚨 PROACTIVE ACCIDENT RISK WARNING SYSTEM (H-CAAR) 🚨")
    print("="*70)
    print("Enter SPATIOTEMPORAL & ENVIRONMENTAL conditions for patrol area:\n")

    speed      = float(input("1. [Location] Speed limit (e.g. 30, 50, 70) : ") or 50)
    hour       = int(input("2. [Time]     Hour of day 0-23 (e.g. 2, 14) : ") or 14)
    is_weekend = int(input("3. [Time]     Weekend? (1: Yes, 0: No)      : ") or 0)
    weather    = input("4. [Weather]  Condition (CLEAR, RAIN, EXTREME): ").upper() or "CLEAR"
    surface    = input("5. [Surface]  Condition (DRY, WET, ICE_SNOW)  : ").upper() or "DRY"
    vehicle_group = input("6. [Vehicle_Group]  Vehicle (Car, Motorcycle, Truck)  : ") or "Car"

    input_data = {
        'Hour'             : hour,
        'Is_Weekend'       : is_weekend,
        'Weather_Group'    : weather,
        'Surface_Group'    : surface,
        'Speed Limit'      : speed,
        'Vehicle_Group'    : vehicle_group,              

        'Light'            : 'DARK' if (hour < 6 or hour > 18) else 'DAYLIGHT', 
        'Route Type'       : 'MARYLAND (STATE)', 
        'Collision Type'   : 'UNKNOWN',          
        'Traffic Control'  : 'NO CONTROLS',
        'Vehicle Movement' : 'MOVING CONSTANT SPEED',
        'Is_Impaired'      : 0,  
        'Is_Distracted'    : 0,
    }

    # Step 3 & 4: Encode and align columns
    input_df      = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Step 5: Run prediction
    raw_prediction = model.predict(input_aligned)[0]
    probabilities  = model.predict_proba(input_aligned)[0]

    p = list(probabilities)
    while len(p) < 3: p.append(0.0)
    prob_severe_pct = p[2] * 100

    # Step 6: Display prediction result
    print("\n" + "-"*70)
    print(" 📊 PREDICTION RESULT & DECISION OVERRIDE")
    print("-"  *70)

    if prob_severe_pct >= 7.0:
        print(f"🛑 RED ALERT TRIGGERED: High risk of severe/fatal injury ({prob_severe_pct:.1f}%)!")
        print("   -> Probability exceeds the 7.0% utility threshold.")
        print("   -> ACTION: Dispatch preventative highway patrol to this corridor.")
    else:
        print(f"✅ NOMINAL CONDITIONS: Severe risk is low ({prob_severe_pct:.1f}% < 7.0% threshold).")
        if raw_prediction == 1:
            print("   -> Default Model Output: CLASS 1 (Safe / Property damage only)")
        else:
            print("   -> Default Model Output: CLASS 2 (Moderate risk)")

    print(f"\n[Raw Probability Matrix]")
    print(f"  Class 1 (Safe)      : {p[0]*100:.1f}%")
    print(f"  Class 2 (Moderate)  : {p[1]*100:.1f}%")
    print(f"  Class 3 (Fatal)     : {p[2]*100:.1f}%")
    print("="*70 + "\n")

if __name__ == "__main__":
    # Step 7: Load model then run prediction loop
    model, model_columns = load_model()
    if model:
        while True:
            predict_m1_severity(model, model_columns)
            cont = input("Test another scenario? (Y/N): ").upper()
            if cont != 'Y':
                break