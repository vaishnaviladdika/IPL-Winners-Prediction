# IPL-Winners-Prediction
Cricket Data Analysis and Match Prediction This project analyzes cricket match data to predict the match winner using Support Vector Machine (SVM) and visualize the teams' performance over the years.It covers the following steps:

# Import Necessary Libraries: 
Utilizes pandas, numpy, matplotlib, seaborn, and scikit-learn.

# Load Dataset:
Reads the cricket data from a CSV file. Ensures the file exists and handles exceptions.

# Data Preparation:
Prints column names for debugging.

Strips spaces from column names.

Converts the 'Match_Winner' column from categorical to numerical values.

Drops columns with more than 50% missing values.

Selects the top two numerical columns as features for model training.

Drops rows with missing values in the selected columns.

# Train-Test Split:
Splits the data into training and testing sets.

# Model Training:
Uses Support Vector Machine (SVM) with a linear kernel to train the model.

Calculates and prints the model accuracy.

# Data Visualization:
Plots the number of matches won by each team per season using a grouped bar chart.

Customizes the plot with titles, labels, legend, and grid for better readability.
