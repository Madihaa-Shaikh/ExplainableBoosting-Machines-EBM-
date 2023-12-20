#!/usr/bin/env python
# coding: utf-8

# In[12]:


#read all data and print all features
import pandas as pd
flight_data = pd.read_csv('aa-delays-2023.csv')
flight_data


#read all data and print all features
import pandas as pd
flight_data = pd.read_csv('aa-delays-2023(02).csv')
# 7 Set thetargetofa delay>15 minutesto1 otherwise to/  Assuming a new binary column 'DELAY_TARGET' to represent delays > 15 minutes
target_feature = 'DEP_DELAY'  # Replace with the actual column name representing delay
threshold = 15  # Set the threshold for delay in minutes

if target_feature in flight_data.columns:
    flight_data['DELAY_TARGET'] = (flight_data[target_feature] > threshold).astype(int)
    print(f"'DELAY_TARGET' column created.")
else:
    print(f"Column '{target_feature}' not found in the dataset. Please verify your data.")


flight_data = flight_data.drop(['FL_DATE','OP_CARRIER','ORIGIN','DEST','Unnamed: 27'], axis=1)
print("'FL_DATE' column dropped.")
flight_data   


# In[41]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Assume you have a DataFrame named 'flight_data'

# Selecting relevant features
selected_features = [
    'CRS_DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF',
    'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'CRS_ELAPSED_TIME',
    'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'CARRIER_DELAY',
    'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
]

# Creating a new DataFrame with only the selected features and target variable
selected_data = flight_data[selected_features + ['DELAY_TARGET']]

# Handling missing values
selected_data.fillna(0, inplace=True)

# Splitting the data into features (X) and target variable (y)
X = selected_data.drop('DELAY_TARGET', axis=1)
y = selected_data['DELAY_TARGET']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train an EBM model using the subset of features and data
ebm_model_subset = ExplainableBoostingClassifier(random_state=42)
ebm_model_subset.fit(X_train, y_train)


# In[61]:


import pandas as pd
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

# Generate synthetic data
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)
flight_data = pd.DataFrame(X, columns=[f"selected_features{i}" for i in range(1, 11)])
flight_data['DELAY_TARGET'] = y

# Split the data into features (X) and target variable (y)
X = flight_data.drop('DELAY_TARGET', axis=1)
y = flight_data['DELAY_TARGET']

# Train an EBM model
ebm_model = ExplainableBoostingClassifier(random_state=42)
ebm_model.fit(X, y)

# Explain global behavior
ebm_global_explanation = ebm_model.explain_global()

# Visualize the global explanation
show(ebm_global_explanation)




# In[66]:


import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from interpret.perf import ROC

# Generate synthetic data
from sklearn.datasets import make_classification

# Generating synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)
flight_data = pd.DataFrame(X, columns=[f"selected_features{i}" for i in range(1, 11)])
flight_data['DELAY_TARGET'] = y

# Split the data into features (X) and target variable (y)
X = flight_data.drop('DELAY_TARGET', axis=1)
y = flight_data['DELAY_TARGET']

# Train an EBM model
ebm_model = ExplainableBoostingClassifier(random_state=42)
ebm_model.fit(X, y)

# 5. Plot ROC and explain it globally
roc_curve = ROC(ebm_model.predict_proba).explain_perf(X, y)
show(roc_curve)

# 6. Plot Scope Rules and explain them
scope_rules = ebm_global_explanation.visualize(inline=True)

# 7. Explain the feature importance and the interactions of two features locally for a special dataset
# Assuming 'feature1' and 'feature2' are two features for which you want local explanations
feature1 = 'DELAY_TARGET'
feature2 = 'DEP_DELAY'
ebm_local_explanation = ebm_global_explanation.explain_local(X[[feature1, feature2]])

# Print or use the local feature importance and interaction values as needed
print("Local Feature Importance:", ebm_local_explanation.get_local_importance_dict())
print("Local Interaction Values:", ebm_local_explanation.get_local_interaction_importance())


# In[67]:


import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from interpret.perf import ROC

# Generate synthetic data
from sklearn.datasets import make_classification

# Generating synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)
flight_data = pd.DataFrame(X, columns=[f"selected_features{i}" for i in range(1, 11)])
flight_data['DELAY_TARGET'] = y

# Split the data into features (X) and target variable (y)
X = flight_data.drop('DELAY_TARGET', axis=1)
y = flight_data['DELAY_TARGET']

# Train an EBM model
ebm_model = ExplainableBoostingClassifier(random_state=42)
ebm_model.fit(X, y)

# 5. Plot ROC and explain it globally
roc_curve = ROC(ebm_model.predict_proba).explain_perf(X, y)
show(roc_curve)

# 6. Plot Scope Rules and explain them
show(ebm_model.explain_global())

# 7. Explain the feature importance and the interactions of two features locally for a special dataset
# Assuming 'feature1' and 'feature2' are two features for which you want local explanations
feature1 = 'selected_features1'
feature2 = 'selected_features2'
ebm_local_explanation = ebm_model.explain_local(X[[feature1, feature2]])

# Print or use the local feature importance and interaction values as needed
print("Local Feature Importance:", ebm_local_explanation.get_local_importance_dict())
print("Local Interaction Values:", ebm_local_explanation.get_local_interaction_importance())


# In[ ]:




