# employee-segmentation-clustering-pca

# Employee Segmentation using Clustering and PCA

## Project Overview
This project demonstrates how to use clustering techniques combined with Principal Component Analysis (PCA) to segment employees based on various performance metrics and training needs. Such segmentation helps in personalizing development plans and identifying high-potential talent within an organization. This project was applied in my current role to enhance employee development and improve organizational performance.

## Objectives
- Segment employees based on performance metrics and training needs.
- Personalize development plans for different employee groups.
- Identify high-potential talent within the organization.
- Showcase practical application of machine learning techniques in a real-world business setting.

## Technologies Used
- Python
- pandas
- scikit-learn
- matplotlib

## Note on Data
**Disclaimer:** The data used in this project has been randomly generated to protect the privacy and confidentiality of actual employee data. This approach ensures that no sensitive information is disclosed while still demonstrating the application of machine learning techniques.

## Steps and Code

### Step 1: Import Libraries and Data
```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Employee data
employees = [
    {"name": "Luis", "position": "Outside Sales"},
    {"name": "Matthew", "position": "Assistant Manager"},
    {"name": "Eddie", "position": "Sales Associate"},
    {"name": "Eduardo", "position": "General Manager"},
    {"name": "Jaime", "position": "Assistant Manager"},
    {"name": "Cesar", "position": "General Manager"},
    {"name": "Gabby", "position": "Sales Associate"},
    {"name": "Sean", "position": "Outside Sales"},
    {"name": "Joe", "position": "Outside Sales"},
    {"name": "Austin", "position": "Sales Associate"},
    {"name": "JJ", "position": "General Manager"},
    {"name": "Campbell", "position": "Sales Associate"},
    {"name": "Tony", "position": "Sales Associate"},
    {"name": "Jarrod", "position": "Sales Associate"},
    {"name": "Jeffrey", "position": "Sales Associate"}
]

# Generate random data
np.random.seed(0)
num_employees = len(employees)
data = {
    "Name": [employee["name"] for employee in employees],
    "Position": [employee["position"] for employee in employees],
    "Customer Satisfaction": np.random.randint(70, 100, num_employees),
    "Performance Ratings": np.random.randint(1, 5, num_employees),
    "Training Hours Completed": np.random.randint(5, 50, num_employees),
    "Scores from Training Assessments": np.random.randint(50, 100, num_employees),
    "Types of Training Completed": np.random.choice(['Sales', 'Customer Service', 'Management', 'Technical'], num_employees),
    "Attendance Records": np.random.randint(90, 100, num_employees),
    "Feedback from Peers and Managers": np.random.choice(['Positive', 'Neutral', 'Negative'], num_employees),
    "Readiness for Managerial Roles": np.random.choice(['High', 'Medium', 'Low'], num_employees),
    "Need for Further Training or Support": np.random.choice(['Yes', 'No'], num_employees),
    "Strengths and Areas for Improvement": np.random.choice(['Strength in Sales', 'Needs Improvement in Technical Skills', 'Good Leadership Qualities', 'Needs Improvement in Customer Service'], num_employees)
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df.head())
```

### Step 2: Preprocess the Data
```python
# Encode categorical columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Extract features and scale them
features = df.drop(columns=['Name', 'Position'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
```

### Step 3: Apply PCA
```python
# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
```

### Step 4: Cluster the Data
```python
# Apply KMeans
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=0)
clusters = kmeans.fit_predict(pca_df)

# Add the cluster information to the DataFrame
pca_df['Cluster'] = clusters
pca_df['Name'] = df['Name']
pca_df['Position'] = df['Position']

# Display the DataFrame
print(pca_df.head())
```

### Step 5: Visualization
```python
# Determine feature influence on each principal component
pca_weights = pd.DataFrame(pca.components_.T, columns=['PCA1', 'PCA2'], index=features.columns)
print(pca_weights)

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Create a scatter plot with a colormap
scatter = plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['Cluster'], cmap='rainbow')

# Add annotations for each employee
for i, txt in enumerate(pca_df['Name']):
    plt.annotate(txt, (pca_df['PCA1'][i], pca_df['PCA2'][i]), fontsize=8)

# Add color bar and labels
colorbar = plt.colorbar(scatter, ticks=[0, 1, 2])
colorbar.set_label('Cluster')
colorbar.set_ticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3'])

# Label the axes with the most influential features
plt.xlabel('Customer Satisfaction, Strengths, and Training Types (PCA1)')
plt.ylabel('Readiness for Managerial Roles and Training Needs (PCA2)')
plt.title('Employee Segmentation using PCA')

# Add a legend to explain clusters
legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.rainbow(i/2), markersize=10) for i in range(3)]
plt.legend(legend_labels, ['Cluster 1', 'Cluster 2', 'Cluster 3'], loc='upper right')

# Show the plot
plt.show()
```

### Interpretation
Cluster 1 (Purple): Employees who need basic training and performance improvement.
Cluster 2 (Green): High performers ready for managerial roles.
Cluster 3 (Red): Employees excelling in current roles but needing targeted training for future responsibilities.

### Conclusion
This project showcases the practical application of machine learning techniques in an organizational setting to improve employee development and performance. The use of PCA and KMeans clustering allows for effective segmentation and targeted training programs, demonstrating my capability to apply ML skills to solve real-world business problems.
