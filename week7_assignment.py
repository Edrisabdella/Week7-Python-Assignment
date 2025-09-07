# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset from sklearn
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    # Display first few rows
    print("First 5 rows of the dataset:")
    print(iris_df.head())
    print("\n")
    
    # Explore dataset structure
    print("Dataset info:")
    print(iris_df.info())
    print("\n")
    
    # Check for missing values
    print("Missing values in each column:")
    print(iris_df.isnull().sum())
    print("\n")
    
    # No missing values in this dataset, but if there were, we would handle them:
    # iris_df = iris_df.dropna()  # or iris_df.fillna(iris_df.mean())
    
except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
# Compute basic statistics
print("Basic statistics of numerical columns:")
print(iris_df.describe())
print("\n")

# Group by species and compute mean of numerical columns
print("Mean values by species:")
species_group = iris_df.groupby('species').mean()
print(species_group)
print("\n")

# Interesting finding
print("Interesting finding: Setosa has significantly smaller petal dimensions compared to other species.")
print("\n")

# Task 3: Data Visualization
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Line chart - Since Iris doesn't have time data, we'll use index as x-axis
# Let's plot the trend of sepal length across observations
axes[0, 0].plot(iris_df.index, iris_df['sepal length (cm)'], label='Sepal Length')
axes[0, 0].plot(iris_df.index, iris_df['petal length (cm)'], label='Petal Length')
axes[0, 0].set_title('Trend of Sepal and Petal Length Across Observations')
axes[0, 0].set_xlabel('Observation Index')
axes[0, 0].set_ylabel('Length (cm)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Bar chart - Comparison of average petal length per species
species_avg = iris_df.groupby('species')['petal length (cm)'].mean()
axes[0, 1].bar(species_avg.index, species_avg.values, color=['skyblue', 'lightgreen', 'lightcoral'])
axes[0, 1].set_title('Average Petal Length by Species')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Average Petal Length (cm)')

# 3. Histogram - Distribution of sepal length
axes[1, 0].hist(iris_df['sepal length (cm)'], bins=15, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Distribution of Sepal Length')
axes[1, 0].set_xlabel('Sepal Length (cm)')
axes[1, 0].set_ylabel('Frequency')

# 4. Scatter plot - Relationship between sepal length and petal length
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species in iris_df['species'].unique():
    species_data = iris_df[iris_df['species'] == species]
    axes[1, 1].scatter(species_data['sepal length (cm)'], 
                       species_data['petal length (cm)'], 
                       label=species, 
                       alpha=0.7)
axes[1, 1].set_title('Sepal Length vs Petal Length')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('iris_analysis.png', dpi=300)
plt.show()

# Additional analysis: Correlation heatmap
plt.figure(figsize=(8, 6))
numeric_df = iris_df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Iris Dataset Features')
plt.tight_layout()
plt.savefig('iris_correlation.png', dpi=300)
plt.show()

print("Analysis complete. Visualizations saved as 'iris_analysis.png' and 'iris_correlation.png'")