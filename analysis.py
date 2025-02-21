import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

# Load dataset
file_path = "F:/Sports/cricket_data.csv"  # Ensure this path is correct

# Check if the file exists
if not os.path.exists(file_path):
    print(f"❌ Error: File not found at {file_path}. Please check the file path and try again.")
    exit()

try:
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Print all column names for debugging
print("Columns in dataset:", df.columns)

# Remove spaces from column names (if any)
df.columns = df.columns.str.strip()

# Check for 'Match_Winner' column
if 'Match_Winner' not in df.columns:
    print("❌ Error: 'Match_Winner' column is missing! Exiting.")
    exit()

# Convert 'Match_Winner' from categorical to numerical values
df['Match_Winner'] = df['Match_Winner'].astype('category').cat.codes

# Drop columns with more than 50% missing values
threshold = 0.5  
df.dropna(thresh=len(df) * threshold, axis=1, inplace=True)

# Select numerical columns (excluding 'Match_Winner')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Match_Winner' in numeric_cols:
    numeric_cols.remove('Match_Winner')

# Ensure there are at least two valid numerical features
if len(numeric_cols) < 2:
    print(f"⚠️ Warning: Not enough numeric columns found! Available: {numeric_cols}")
    exit()

# Select the top two numerical columns
features = numeric_cols[:2]
target = 'Match_Winner'

# Drop rows with missing values in selected columns
df.dropna(subset=features + [target], inplace=True)

# Ensure sufficient data for train-test split
if df.shape[0] < 2:
    print("❌ Error: Not enough data points after cleaning. Exiting.")
    exit()

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train a Machine Learning model
#model = RandomForestClassifier(n_estimators=100, random_state=42)
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Predict outcomes
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Data Visualization: Yearly Teams Participation
plt.figure(figsize=(12, 6))

# Count number of matches won by each team per season
team_wins_by_year = df.groupby(['season', 'Match_Winner']).size().unstack()

# Plot grouped bar chart
team_wins_by_year.plot(kind='bar', stacked=False, figsize=(12, 6), colormap='viridis')
#random forest data visualization
#plt.figure(figsize=(10, 5))
#sns.barplot(x=df['Match_Winner'].value_counts().index, y=df['Match_Winner'].value_counts().values)
#plt.title("Match Winner Distribution")
#plt.xlabel("Team")
#plt.ylabel("Win Count")
#plt.xticks(rotation=90)
#plt.show()
#svm data visualization
plt.title("🏏 Match Wins by Teams Over the Years")
plt.xlabel("Year (Season)")
plt.ylabel("Number of Wins")
plt.legend(title="Teams", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


