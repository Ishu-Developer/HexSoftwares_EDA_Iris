# ======================================
# PROJECT 2: Data Exploration on Iris Dataset
# @HexSoftwares Internship — Ishu
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. Load Iris Dataset

df = sns.load_dataset('iris') # columns: sepal_length, sepal_width, petal_length, petal_width,species
print("Dataset Shape :",df.shape)
print("\nFirst 5 Rows :",df.head())
print("\nColumn Names :",df.columns.tolist())
print("\nData Types :",df.dtypes)

# 2. Missing Values Check & Handling

print("\nMissing Values Count :",df.isnull().sum())
print("\nMissing Values Percentage :")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent.round(2))

if df.isnull().sum().any():
    # Numerical columns -> fill with median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categorical columns -> fill with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\nAfter Cleaning — Missing Values :",df.isnull().sum())
else:
    print("\nNo missing values found in the Iris dataset.")

# 3. Basic Statistical Analysis

print("\nBasic Statistics (Numerical Features) :",df.describe())  
print("\nClass Distribution (Species) :",df['species'].value_counts())
print("\nMean of Features by Species :",df.groupby('species').mean().round(2))

# 4. Visualizations 

sns.set_theme(style="whitegrid", palette="muted")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Iris Dataset — Data Exploration', fontsize=18, fontweight='bold', y=0.98)

# Sepal Length distribution 
sns.histplot(
    data=df, x='sepal_length', kde=True,
    bins=20, color='#4f98a3', ax=axes[0, 0]
)
axes[0, 0].set_title('Sepal Length Distribution')

# Sepal Width distribution 
sns.histplot(
    data=df, x='sepal_width', kde=True,
    bins=20, color='#e05c5c', ax=axes[0, 1]
)
axes[0, 1].set_title('Sepal Width Distribution')

# Petal Length distribution 
sns.histplot(
    data=df, x='petal_length', kde=True,
    bins=20, color='#7a68a6', ax=axes[0, 2]
)
axes[0, 2].set_title('Petal Length Distribution')

# Petal Width distribution
sns.histplot(
    data=df, x='petal_width', kde=True,
    bins=20, color='#8fbf4d', ax=axes[1, 0]
)
axes[1, 0].set_title('Petal Width Distribution')

# Species pie chart (class distribution) 
species_counts = df['species'].value_counts()
axes[1, 1].pie(
    species_counts.values,
    labels=species_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#4f98a3', '#e05c5c', '#8fbf4d']
)
axes[1, 1].set_title('Species Distribution (Pie Chart)')

# Species Pie Chart (average petal length share)
petal_mean = df.groupby('species')['petal_length'].mean()
axes[1, 2].pie(
    petal_mean.values,
    labels=petal_mean.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#4f98a3', '#e05c5c', '#8fbf4d']
)
axes[1, 2].set_title('Average Petal Length Share by Species')

plt.tight_layout()
plt.savefig('iris_exploration_charts.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Combined Iris EDA figure saved as: iris_exploration_charts.png")

pair = sns.pairplot(df, hue='species', diag_kind='hist')
pair.fig.suptitle('Pairplot of Iris Features by Species', y=1.02)
pair.savefig('iris_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Pairplot saved as: iris_pairplot.png")