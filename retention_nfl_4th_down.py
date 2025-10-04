import pandas as pd
import nfl_data_py as nfl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# -----------------------
# 1. Load play-by-play data
# -----------------------
seasons = [2018, 2019, 2020, 2021, 2022]
pbp = nfl.import_pbp_data(seasons, downcast=True)

# -----------------------
# 2. 4th-down conversion model
# -----------------------
fourth_go = pbp[(pbp['down'] == 4) & (pbp['play_type'].isin(['pass','run']))].copy()
fourth_go['success'] = fourth_go['yards_gained'] >= fourth_go['ydstogo']

features = ['ydstogo','yardline_100']
X = fourth_go[features].fillna(0)
y = fourth_go['success'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

conv_model = LogisticRegression()
conv_model.fit(X_train, y_train)
print("4th-down conversion accuracy:", conv_model.score(X_test, y_test))

# Save conversion model
joblib.dump(conv_model, "conv_model.pkl")

# -----------------------
# 3. Field goal model
# -----------------------
fg_plays = pbp[(pbp['play_type']=='field_goal') & (pbp['field_goal_result'].notnull())].copy()
fg_plays['make'] = fg_plays['field_goal_result'] == 'made'

fg_X = fg_plays[['kick_distance']].fillna(0)
fg_y = fg_plays['make'].astype(int)

fg_model = LogisticRegression()
fg_model.fit(fg_X, fg_y)

# Save field goal model
joblib.dump(fg_model, "fg_model.pkl")

print("Models trained and saved successfully.")
