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
    print("\n" + "="*65)
    print(" ACCIDENT RISK WARNING SYSTEM ")
    print("="*65)
    print("Please enter the context (Time - Environment - Location - Driver):\n")

    speed        = float(input("1.  [Location]    Speed limit (e.g. 30, 50, 70)                      : ") or 50)
    hour         = int(input("2.  [Time]        Hour of day 0-23 (e.g. 14)                         : ") or 14)
    is_weekend   = int(input("3.  [Time]        Weekend? (1: Yes, 0: No)                           : ") or 0)

    weather      = input("4.  [Environment] Weather     (e.g. CLEAR, RAIN, EXTREME)            : ").upper() or "CLEAR"
    light        = input("5.  [Environment] Light       (e.g. DAYLIGHT, DARK)                  : ").upper() or "DAYLIGHT"
    surface      = input("6.  [Environment] Surface     (e.g. DRY, WET, ICE_SNOW)              : ").upper() or "DRY"

    route        = input("7.  [Location]    Route type  (e.g. County, Maryland)                : ") or "UNKNOWN"
    collision    = input("8.  [Collision]   Collision type (e.g. HEAD ON, REAR END)            : ").upper() or "UNKNOWN"
    vehicle      = input("9.  [Vehicle]     Vehicle group (e.g. Car, Motorcycle, Truck)        : ") or "Car"
    traffic_ctrl = input("10. [Infrastructure] Traffic control (e.g. TRAFFIC SIGNAL, STOP SIGN): ").upper() or "NO CONTROLS"
    movement     = input("11. [Vehicle]     Movement    (e.g. MOVING CONSTANT SPEED)           : ").upper() or "UNKNOWN"

    is_impaired  = int(input("12. [Driver]      Impaired (alcohol/substance)? (1: Yes, 0: No)      : ") or 0)
    is_distracted= int(input("13. [Driver]      Distracted? (1: Yes, 0: No)                        : ") or 0)

    # Step 3: Package input data with correct column names
    input_data = {
        'Hour'             : hour,
        'Is_Weekend'       : is_weekend,
        'Weather_Group'    : weather,
        'Light'            : light,
        'Surface_Group'    : surface,
        'Route Type'       : route,
        'Speed Limit'      : speed,
        'Collision Type'   : collision,
        'Vehicle_Group'    : vehicle,
        'Traffic Control'  : traffic_ctrl,
        'Vehicle Movement' : movement,
        'Is_Impaired'      : is_impaired,
        'Is_Distracted'    : is_distracted,
    }

    # Step 4: Encode and align columns to match training schema
    input_df      = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Step 5: Run prediction and extract probabilities
    raw_prediction = model.predict(input_aligned)[0]
    probabilities  = model.predict_proba(input_aligned)[0]

    p = list(probabilities)
    while len(p) < 3:
        p.append(0.0)

    prob_severe_pct = p[2] * 100

    # Step 6: Display prediction result
    print("\n" + "-"*65)
    print(" PREDICTION RESULT & ANALYSIS")
    print("-"*65)

    if prob_severe_pct >= 6.0:
        print(f"RED ALERT: High risk of severe injury ({prob_severe_pct:.1f}%)!")
        print("Probability of a dangerous accident has reached 6% threshold — PRIORITY ACTION RECOMMENDED.")
    else:
        if raw_prediction == 1:
            print("RESULT: CLASS 1 (Low risk — mainly property damage)")
        elif raw_prediction == 2:
            print("RESULT: CLASS 2 (Moderate risk — possible minor/moderate injury)")
        else:
            print("RESULT: CLASS 3 (High risk — severe injury)")

    print(f"\n[Detailed probability breakdown]")
    print(f"  Class 1 (Low risk)      : {p[0]*100:.1f}%")
    print(f"  Class 2 (Moderate risk) : {p[1]*100:.1f}%")
    print(f"  Class 3 (High risk)     : {p[2]*100:.1f}%")
    print("="*65 + "\n")
 
if __name__ == "__main__":
    # Step 7: Load model then run prediction loop
    model, model_columns = load_model()
    if model:
        while True:
            predict_m1_severity(model, model_columns)
            cont = input("Test another scenario? (Y/N): ").upper()
            if cont != 'Y':
                break
