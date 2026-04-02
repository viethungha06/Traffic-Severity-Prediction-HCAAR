import pandas as pd
import sqlite3
import os
import warnings
warnings.filterwarnings('ignore')

def execute_query(query_name, query, conn, file_name=None):
    print(f"\n{query_name}")
    try:
        result = pd.read_sql_query(query, conn)
        print(result.to_string(index=False))
        if file_name:
            export_dir = 'output/sql_results'
            os.makedirs(export_dir, exist_ok=True)
            export_path = os.path.join(export_dir, f"{file_name}.csv")
            result.to_csv(export_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"[!] Query error: {e}")
    print("-" * 80)

def run_sql_analysis():
    csv_path = 'output/cleaned_dataset.csv'
    db_name = 'output/traffic_data.db' 
    
    # Step 1: Check if cleaned dataset exists
    if not os.path.exists(csv_path):
        print("[!] Please run cleaning.py first!")
        return
        
    # Step 2: Load CSV into SQLite database
    print(f"[*] Loading data into database {db_name}...")
    os.makedirs('output', exist_ok=True)
    
    df = pd.read_csv(csv_path, low_memory=False)
    conn = sqlite3.connect(db_name)
    df.to_sql('accidents', conn, if_exists='replace', index=False, chunksize=10000)

    # --- 1. INJURY SEVERITY ---
    q1 = """
        SELECT "Injury Severity", Severity_Class, COUNT(*) as Total 
        FROM accidents 
        GROUP BY "Injury Severity", Severity_Class 
        ORDER BY Total DESC;
    """
    execute_query("--- 1. INJURY SEVERITY ---", q1, conn, "Q1_Injury_Severity")

    # --- 2. SURFACE CONDITION ---
    q2 = """
        SELECT Surface_Group, COUNT(*) as Total 
        FROM accidents 
        GROUP BY Surface_Group 
        ORDER BY Total DESC;
    """
    execute_query("--- 2. SURFACE CONDITION ---", q2, conn, "Q2_Surface_Condition")

    # --- 3. SEVERE COLLISION TYPES (Level 3 is the highest) ---
    q3 = """
        SELECT "Collision Type", COUNT(*) as Severe_Cases 
        FROM accidents 
        WHERE Severity_Class = 3 AND "Collision Type" IS NOT NULL 
        GROUP BY "Collision Type" 
        ORDER BY Severe_Cases DESC 
        LIMIT 5;
    """
    execute_query("--- 3. SEVERE COLLISION TYPES ---", q3, conn, "Q3_Collision_Type")

    # --- 4. YEARLY TREND (Level 3 updated) ---
    q4 = """
        SELECT Year, COUNT(*) as Total, 
               SUM(CASE WHEN Severity_Class = 3 THEN 1 ELSE 0 END) as Severe 
        FROM accidents 
        GROUP BY Year 
        ORDER BY Year ASC;
    """
    execute_query("--- 4. YEARLY TREND ---", q4, conn, "Q4_Yearly_Trend")

    # --- 5. DAY VS NIGHT ---
    q5 = """
        SELECT 
            CASE 
                WHEN Time_of_Day IN ('Morning', 'Afternoon') THEN 'Daytime' 
                ELSE 'Nighttime' 
            END as Period, 
            COUNT(*) as Total, 
            ROUND(AVG(Severity_Class), 2) as Avg_Severity 
        FROM accidents 
        GROUP BY Period;
    """
    execute_query("--- 5. DAY VS NIGHT ---", q5, conn, "Q5_Day_Night")

    # --- 6. LICENSE STATE ---
    q6 = """
        SELECT "Drivers License State", COUNT(*) as Total 
        FROM accidents 
        WHERE "Drivers License State" IS NOT NULL 
          AND "Drivers License State" != 'Unknown' 
        GROUP BY "Drivers License State" 
        ORDER BY Total DESC 
        LIMIT 10;
    """
    execute_query("--- 6. LICENSE STATE ---", q6, conn, "Q6_License_State")

    # --- 7. HIGH-RISK TIME SLOTS ---
    q7 = """
        SELECT Day_of_Week, Hour, COUNT(*) as Total 
        FROM accidents 
        GROUP BY Day_of_Week, Hour 
        ORDER BY Total DESC 
        LIMIT 10;
    """
    execute_query("--- 7. HIGH-RISK TIME SLOTS ---", q7, conn, "Q7_Time_Slots")

    # --- 8. HIGH-RISK ROUTE TYPES ---
    q8 = """
        SELECT "Route Type", COUNT(*) as Total, 
               ROUND(AVG(Severity_Class), 2) as Avg_Severity 
        FROM accidents 
        WHERE "Route Type" IS NOT NULL AND "Route Type" != '' 
        GROUP BY "Route Type" 
        ORDER BY Total DESC 
        LIMIT 10;
    """
    execute_query("--- 8. HIGH-RISK ROUTE TYPES ---", q8, conn, "Q8_Route_Type")

    # --- 9. WEATHER & LIGHT CONDITIONS ---
    q9 = """
        SELECT Weather_Group, Light, 
               ROUND(AVG(Severity_Class), 2) as Avg_Severity, 
               COUNT(*) as Total 
        FROM accidents 
        WHERE Weather_Group != 'UNKNOWN' AND Light != 'UNKNOWN' 
        GROUP BY Weather_Group, Light 
        HAVING Total > 50 
        ORDER BY Avg_Severity DESC 
        LIMIT 10;
    """
    execute_query("--- 9. WEATHER & LIGHT CONDITIONS ---", q9, conn, "Q9_Weather_Light")

    # --- 10. SPEED LIMIT --- 
    q10 = """
        SELECT 
            CASE 
                WHEN "Speed Limit" <= 30 THEN '0-30 mph' 
                WHEN "Speed Limit" <= 50 THEN '31-50 mph' 
                ELSE '> 50 mph' 
            END as Speed_Group, 
            COUNT(*) as Total, 
            ROUND(AVG(Severity_Class), 2) as Avg_Severity 
        FROM accidents 
        WHERE "Speed Limit" IS NOT NULL 
        GROUP BY Speed_Group 
        ORDER BY Avg_Severity DESC;
    """
    execute_query("--- 10. SPEED LIMIT ---", q10, conn, "Q10_Speed_Limit")

    # Step 3: Close database connection
    conn.close()
    print("\n[V] All SQL queries completed!")

if __name__ == "__main__":
    run_sql_analysis()
