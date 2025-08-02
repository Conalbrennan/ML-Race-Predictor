import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from scripts.load_race_data import get_austria_race_data
from scripts.driver_trends import get_driver_stats_and_trends

TARGET_YEAR = 2025
TARGET_RACE_NAME = 'Austria'
TARGET_ROUND = 11  # e.g. Austria is Round 11 in the calendar

# Load 2024 Austria race data for training 
df = get_austria_race_data(2024)
df['Compound'] = LabelEncoder().fit_transform(df['Compound'])

# Load 2025 qualifying times 
quali_df = pd.read_csv('data/austria_2025_quali.csv')

# Add driver trend features from previous races 
trend_df = get_driver_stats_and_trends(TARGET_YEAR, TARGET_ROUND)
quali_df = quali_df.merge(trend_df, on='Driver', how='left')

# Separate known and new drivers 
known_drivers_df = quali_df[quali_df['Driver'].isin(df['Driver'])]
new_drivers_df = quali_df[~quali_df['Driver'].isin(df['Driver'])]

# Merge training features 
merged_df = df.merge(known_drivers_df, on='Driver', how='inner')

# Prepare training set 
merged_df = merged_df.sort_values('QualiTime')
merged_df['Rank'] = range(1, len(merged_df) + 1)

features = [
    'FastestLap', 'Compound', 'DNF', 'QualiTime',
    'avg_qualifying_time', 'avg_finish_position', 'dnf_rate'
]

X_known = merged_df[features]
y_known = merged_df['Rank']

# Train model 
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_known, y_known)
merged_df['PredictedPosition'] = model.predict(X_known)

# Predict for new drivers using fallback values 
avg_fastest_lap = df['FastestLap'].mean()
avg_compound = df['Compound'].mode()[0]
avg_dnf = 0

new_rows = []
for _, row in new_drivers_df.iterrows():
    new_rows.append({
        'Driver': row['Driver'],
        'FastestLap': avg_fastest_lap,
        'Compound': avg_compound,
        'DNF': avg_dnf,
        'QualiTime': row['QualiTime'],
        'avg_qualifying_time': row['avg_qualifying_time'],
        'avg_finish_position': row['avg_finish_position'],
        'dnf_rate': row['dnf_rate']
    })

new_df = pd.DataFrame(new_rows)
X_new = new_df[features]
new_df['PredictedPosition'] = model.predict(X_new)

# Combine and show top 10 by predicted finishing position
final_df = pd.concat([
    merged_df[['Driver', 'FastestLap', 'PredictedPosition']],
    new_df[['Driver', 'FastestLap', 'PredictedPosition']]
])

final_df = final_df.sort_values('PredictedPosition').reset_index(drop=True)
top_10 = final_df.head(10)

print("\n🏁 Predicted Top 10 for 2025 Austria GP (based on finishing position):")
print(top_10[['Driver', 'FastestLap']])
