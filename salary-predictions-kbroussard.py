# 📌 Post-College Salary Prediction: Data Exploration & Visualization

# **Author:** Kersha Broussard
# **Dataset:** Post-College Salaries (Kaggle)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
csv_path = r"C:\Users\kbrou\Downloads\post_college_salaries\final-post-college-salaries.csv"
df = pd.read_csv(csv_path)

# Data Cleaning
# Convert salary columns to numeric (remove "$" and ",")
df["Early Career Pay"] = df["Early Career Pay"].replace('[\$,]', '', regex=True).astype(float)
df["Mid-Career Pay"] = df["Mid-Career Pay"].replace('[\$,]', '', regex=True).astype(float)

# Convert "% High Meaning" to numeric
df["% High Meaning"] = df["% High Meaning"].str.replace('%', '', regex=True)
df["% High Meaning"] = pd.to_numeric(df["% High Meaning"], errors="coerce")

# Drop missing values
df.dropna(inplace=True)

# **1. Histograms for Feature Distributions**
plt.figure(figsize=(12, 6))
df[["Early Career Pay", "Mid-Career Pay"]].hist(bins=30, figsize=(12, 6))
plt.suptitle("Distribution of Early Career Pay and Mid-Career Pay")
plt.show()

# **2. Boxplot for Mid-Career Pay by Major**
plt.figure(figsize=(15, 6))
sns.boxplot(y=df["Mid-Career Pay"], x=df["Major"], data=df)
plt.xticks(rotation=90)
plt.title("Mid-Career Pay Distribution by Major")
plt.xlabel("Major")
plt.ylabel("Mid-Career Pay ($)")
plt.show()

# **3. Scatterplot: Early vs. Mid-Career Pay**
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["Early Career Pay"], y=df["Mid-Career Pay"])
plt.title("Early Career Pay vs Mid-Career Pay")
plt.xlabel("Early Career Pay ($)")
plt.ylabel("Mid-Career Pay ($)")
plt.show()

# **4. Correlation Heatmap**
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlations with Mid-Career Pay")
plt.show()

# Save the processed dataset for future modeling
df.to_csv("processed_post_college_salaries.csv", index=False)

print("✅ Data visualization complete!")
