import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Ρυθμίσεις εμφάνισης
sns.set(style="whitegrid")

print("--- ΕΝΑΡΞΗ ΨΗΦΙΑΚΟΥ ΔΙΔΥΜΟΥ ---")


# ΦΑΣΗ 1: Φόρτωση και Προετοιμασία

filename = '2019Floor7.csv'
print(f"1. Φόρτωση αρχείου: {filename} ...")

try:
    df = pd.read_csv(filename)
    df['Datetime'] = pd.to_datetime(df['Date'])
    df = df.set_index('Datetime')

    room_df = pd.DataFrame()
    room_df['Energy'] = df['z2_AC1(kW)'] + df['z2_Light(kW)'] + df['z2_Plug(kW)']
    room_df['Temperature'] = df['z2_S1(degC)']    
    room_df['Light_Lux'] = df['z2_S1(lux)']        
    room_df['Lighting_Power'] = df['z2_Light(kW)']
    room_df['Hour'] = room_df.index.hour
    room_df = room_df.dropna()
    print(f"({len(room_df)} μετρήσεις)")

    
    # ΕΥΡΕΣΗ ΜΙΑΣ ΤΥΠΙΚΗΣ ΕΒΔΟΜΑΔΑΣ 
    
    first_monday_idx = room_df[room_df.index.dayofweek == 0].index[0]
    week_start = room_df.index.get_loc(first_monday_idx)
    week_end = week_start + (5 * 24 * 60)
    
    weekly_data = room_df.iloc[week_start:week_end]
    
    # Resample για τα γραφήματα της εβδομάδας για να είναι καθαρά
    weekly_data_clean = weekly_data.resample('30min').mean()

    # ΓΡΑΦΗΜΑ 1: ΤΥΠΙΚΗ ΕΒΔΟΜΑΔΑ
    print("\n[ΓΡΑΦΗΜΑ 1/3] Εμφάνιση Τυπικής Εβδομάδας")
    plt.figure(figsize=(14, 6))
    plt.plot(weekly_data_clean.index, weekly_data_clean['Energy'], label='Πραγματική Κατανάλωση', color='#1f77b4', linewidth=2)
    plt.title('Ανάλυση Δεδομένων: Προφίλ Τυπικής Εβδομάδας (Δευ-Παρ)', fontsize=16)
    plt.xlabel('Ημέρα', fontsize=12)
    plt.ylabel('Ενέργεια (kW)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    # ΦΑΣΗ 2: Εκπαίδευση
    
    print("\n2. Εκπαίδευση Μοντέλου ")
    X = room_df[['Temperature', 'Light_Lux', 'Lighting_Power', 'Hour']]
    y = room_df['Energy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    
    # ΦΑΣΗ 3: Αξιολόγηση
    
    print("\n3. Αξιολόγηση Ακρίβειας")
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    print(f"Ακρίβεια (R2 Score): {score:.4f}")

    # ΓΡΑΦΗΜΑ 2: ΑΚΡΙΒΕΙΑ 
    print("\n[ΓΡΑΦΗΜΑ 2/3] Επικύρωση ακρίβειας Μοντέλου")
    X_week = weekly_data[['Temperature', 'Light_Lux', 'Lighting_Power', 'Hour']]
    y_week_actual = weekly_data['Energy']
    y_week_pred = model.predict(X_week)
    
    comparison_df = pd.DataFrame({'Actual': y_week_actual, 'Predicted': y_week_pred}, index=weekly_data.index)
    comparison_clean = comparison_df.resample('30min').mean()
    
    plt.figure(figsize=(14, 6))
    plt.plot(comparison_clean.index, comparison_clean['Actual'], label='Πραγματική Τιμή', color='blue', linewidth=2, alpha=0.6)
    plt.plot(comparison_clean.index, comparison_clean['Predicted'], label='Πρόβλεψη Digital Twin', color='orange', linestyle='--', linewidth=2)
    plt.title(f'Επικύρωση Μοντέλου: Ακρίβεια Πρόβλεψης (R2: {score:.3f})', fontsize=16)
    plt.xlabel('Ημέρα', fontsize=12)
    plt.ylabel('Ενέργεια (kW)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    # ΦΑΣΗ 4: PEAK DAY 
    
    # Βρίσκουμε τη στιγμή της αιχμής 
    peak_moment = room_df['Energy'].idxmax()
    
    
    # Παίρνουμε δεδομένα  12 ώρες
    center_loc = room_df.index.get_loc(peak_moment)
    start_loc = max(0, center_loc - 720)
    end_loc = min(len(room_df), center_loc + 720)
    
    target_day_X = X.iloc[start_loc:end_loc].copy()
    
    # 1. Baseline
    current_consumption = model.predict(target_day_X)
    
  # 2. Smart Eco (Υβριδικό Σενάριο)
    optimized_scenario = target_day_X.copy()
    
    # ΒΗΜΑ 1: AI - Daylight Harvesting (Φώτα)
    # Το model καταλαβαίνει τη σχέση Lux και Φώτων
    mask_daylight = optimized_scenario['Light_Lux'] > 300
    optimized_scenario.loc[mask_daylight, 'Lighting_Power'] = optimized_scenario.loc[mask_daylight, 'Lighting_Power'] * 0.20
    
    # Παίρνουμε την πρόβλεψη του AI με τα μειωμένα φώτα
    prediction_with_smart_lights = model.predict(optimized_scenario)
    
    # ΒΗΜΑ 2:Setpoint Adjustment (+2.5°C)
    # Επειδή το μοντέλο μπερδεύει την αύξηση θερμοκρασίας με τον καύσωνα,
    
    # Κανόνας: +1°C  = ~4.6% εξοικονόμηση στην ψύξη.
    # Άρα +2.5°C einai 10% εξοικονόμηση
    
    thermal_savings_factor = 0.885 
    new_consumption = prediction_with_smart_lights * thermal_savings_factor
    
    # KPIs veltiwsh
    total_current = current_consumption.sum()
    total_new = new_consumption.sum()
    savings_pct = ((total_current - total_new) / total_current) * 100
    
    print(f"\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ (PEAK DAY SCENARIO) ---")
    print(f"Συνολική Ενέργεια (Baseline):  {total_current:.2f} kWh")
    print(f"Συνολική Ενέργεια (Smart Eco): {total_new:.2f} kWh")
    print(f"Εξοικονόμηση : {savings_pct:.2f}%")
    
    #  ΓΡΑΦΗΜΑ 3: peak day  
    print("\n[ΓΡΑΦΗΜΑ 3/3] Εμφάνιση Peak Day (Λεπτομερής)...")
    plt.figure(figsize=(14, 6))
    
    plt.plot(target_day_X.index, current_consumption, label='Baseline (High Peak)', color='#d62728', linewidth=2)
    plt.plot(target_day_X.index, new_consumption, label='Smart Eco (Optimized)', color='#2ca02c', linewidth=2)
    
    
    plt.title('Digital Twin: Διαχείριση Φορτίου σε Ημέρα Αιχμής (Peak Day)', fontsize=16)
    plt.xlabel('Ώρα', fontsize=12)
    plt.ylabel('Ενέργεια (kW)', fontsize=12)
    
    plt.fill_between(target_day_X.index, current_consumption, new_consumption, color='#2ca02c', alpha=0.2, label='Εξοικονόμηση')
    plt.legend()
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\nΣΦΑΛΜΑ: {e}")