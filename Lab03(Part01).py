#!/usr/bin/env python
# coding: utf-8

# In[13]:


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




# In[51]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have a DataFrame named 'flight_data'

# Print the list of columns in your DataFrame
print("Columns in the DataFrame:", list(flight_data.columns))

# Specify categorical columns (adjust based on your dataset)
categorical_columns = ['OP_CARRIER_FL_NUM', 'CANCELLATION_CODE']  # Add your categorical columns here

# Verify if categorical columns are present in the DataFrame
for col in categorical_columns:
    if col not in flight_data.columns:
        raise ValueError(f"'{col}' column is not present in the DataFrame. Choose a valid categorical column.")

# Create features (X) and target variable (y)
X = flight_data.drop('DELAY_TARGET', axis=1)
y = flight_data['DELAY_TARGET']

# Map delays to binary values (0: not delayed, 1: delayed)
y_binary = y.apply(lambda x: 1 if x > 0 else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Create a ColumnTransformer to apply one-hot encoding to categorical columns
# Create a ColumnTransformer to apply one-hot encoding to categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'  # Pass through other columns as is
)

# Apply the preprocessing to the features
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Apply the preprocessing to the features
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Create a decision tree classifier with a depth of 3
dt_classifier = DecisionTreeClassifier(max_depth=3)

# Fit the model
dt_classifier.fit(X_train_encoded, y_train)

# Plot the decision tree graphically
# Plot the decision tree graphically
plt.figure(figsize=(15, 10))
plot_tree(
    dt_classifier,
    filled=True,
    feature_names=list(preprocessor.get_feature_names_out(X_train.columns)),
    class_names=['Not Delayed', 'Delayed']
)
plt.show()


# In[30]:


from sklearn.tree import export_text

# Print the text representation of the decision tree
tree_rules = export_text(dt_classifier, feature_names=list(preprocessor.get_feature_names_out(X_train.columns)))
print(tree_rules)


# In[ ]:

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


# In[ ]:

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

# 6. Explain global feature importance
global_explanation = ebm_model.explain_global()
show(global_explanation)




# In[ ]:




