# 1. Data Loading and Exploration
## 1.1 Load the dataset
import pandas as pd
import numpy as np

dataset = pd.read_csv('anime_list_encoded.csv')

print(dataset)
## 1.2 Basic statistics and data overview
print(dataset.describe())
## 1.3 Data cleaning (handle missing values, outliers, etc.)
dataset['url'].fillna('URL_NOT_AVAILABLE', inplace=True)

numerical_columns = ['score']
for col in numerical_columns:
    dataset[col].fillna(dataset[col].mean(), inplace=True)

categorical_columns = ['jpName', 'lastUpdate', 'engName', 'source', 'status', 'aired', 'duration', 'rating', 'studios', 'genres', 'producer', 'licensors']
for col in categorical_columns:
    dataset[col].fillna('NOT_AVAILABLE', inplace=True)

dataset['missing_data'] = dataset.isnull().any(axis=1).astype(int)

# print count of datapoints with value nan
print(dataset.isnull().sum())
# 2. Content-based Filtering
## 2.1 Feature extraction and preprocessing
from sklearn.preprocessing import LabelEncoder


labelencoder = LabelEncoder()
dataset['rating'] = labelencoder.fit_transform(dataset['rating'])


print(dataset['rating'].describe())

# feasture selection
#  which features are important for our model?
#  which features are not important for our model?

# 1. correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
# plot heat map
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# 2. feature importance
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(dataset.drop(['score'], axis=1),dataset['score'])
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=dataset.drop(['score'], axis=1).columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()




## 2.2 Compute similarity scores between shows

## 2.3 Generate recommendations based on similarity
# 3. Collaborative Filtering
## 3.1 Prepare user-item interaction matrix/data
## 3.2 Implement User-User Collaborative Filtering
## 3.3 Implement Item-Item Collaborative Filtering
# 4. Evaluation
## 4.1 Split data into training and test sets
## 4.2 Define evaluation metrics
## 4.3 Evaluate the performance of recommendation models
# 5. Iteration and Improvement
## 5.1 Analyze evaluation results
## 5.2 Refine models or features based on feedback
## 5.3 Experiment with hybrid methods or other algorithms
# 6. Scalability Considerations
## 6.1 Optimize data structures for faster computation
## 6.2 Consider using specialized libraries or tools
## 6.3 Think about deployment and serving recommendations in real-time
# 7. User Feedback Loop
## 7.1 Implement user feedback mechanisms (like/dislike, ratings)
## 7.2 Adjust recommendation logic based on user feedback
## 7.3 Monitor and continuously improve recommendation quality