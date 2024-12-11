#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('kaggle_survey_2022_responses.csv', low_memory=False)
print(df.head())


# In[4]:


import matplotlib.pyplot as plt

# Ensure the column name matches
program_language = [f"Q12_{i}" for i in range(1, 16)]  # Adjust range to match your column numbers
language_data = df[program_language]

# Combine all the columns into one series, ignoring NaN values
all_languages = language_data.stack().reset_index(drop=True)

# Count occurrences of each programming language
language_counts = all_languages.value_counts()

# Plot the top 10 programming languages
language_counts.head(10).plot(kind='barh', figsize=(10, 6))
plt.title('Top 10 Most Common Programming Languages')
plt.xlabel('Number of Respondents')
plt.ylabel('Programming Languages')
plt.gca().invert_yaxis()  # Show the highest bar on top
plt.show()


# In[12]:


# Extract the relevant columns (Q13_1 through Q13_14)
ide_columns = [f"Q13_{i}" for i in range(1, 15)]  # Adjust range to match your column numbers
ide_data = df[ide_columns]

# Combine all the columns into one series, ignoring NaN values
all_ide_usage = ide_data.stack().reset_index(drop=True)

# Count occurrences of each IDE
ide_counts = all_ide_usage.value_counts()

# Plot the top 10 most used IDEs
ide_counts.head(10).plot(kind='barh', figsize=(10, 6))
plt.title('Top 10 Most Used Integrated Development Environments (IDEs)')
plt.xlabel('Number of Respondents')
plt.ylabel('Integrated Development Environments')
plt.gca().invert_yaxis()  # Show the highest bar on top
plt.show()


# In[13]:


# Extract the relevant columns for hosted notebooks (Q14_1 through Q14_16)
notebook_columns = [f"Q14_{i}" for i in range(1, 17)]  # Adjust range to match your dataset
notebooks_used_data = df[notebook_columns]

# Combine all the columns into one series, ignoring NaN values
all_notebooks_used = notebooks_used_data.stack().reset_index(drop=True)

# Count occurrences of each notebook usage
notebooks_used_counts = all_notebooks_used.value_counts()

# Plot the top 10 most used hosted notebooks
notebooks_used_counts.head(10).plot(kind='barh', figsize=(10, 6))
plt.title('Top 10 Most Used Hosted Notebooks')
plt.xlabel('Number of Respondents')
plt.ylabel('Hosted Notebooks')
plt.gca().invert_yaxis()  # Show the highest bar on top
plt.show()


# In[14]:


# Extract the relevant columns for data visualization libraries (Q15_1 through Q15_15)
data_viz_columns = [f"Q15_{i}" for i in range(1, 16)]  # Adjust range if needed
data_viz_used_data = df[data_viz_columns]

# Combine all the columns into one series, ignoring NaN values
all_data_viz_used = data_viz_used_data.stack().reset_index(drop=True)

# Count occurrences of each data visualization library
data_viz_used_counts = all_data_viz_used.value_counts()

# Plot the top 10 most used data visualization libraries
data_viz_used_counts.head(10).plot(kind='barh', figsize=(10, 6))
plt.title('Top 10 Most Used Data Visualization Libraries')
plt.xlabel('Number of Respondents')
plt.ylabel('Data Visualization Libraries')
plt.gca().invert_yaxis()  # Show the highest bar on top
plt.show()


# In[15]:


# Extract the relevant columns for machine learning frameworks (Q17_1 through Q17_15)
ml_framework_columns = [f"Q17_{i}" for i in range(1, 16)]  # Adjust range if needed
ml_frameworks_data = df[ml_framework_columns]

# Combine all the columns into one series, ignoring NaN values
all_ml_frameworks_used = ml_frameworks_data.stack().reset_index(drop=True)

# Count occurrences of each machine learning framework
ml_frameworks_counts = all_ml_frameworks_used.value_counts()

# Plot the top 10 most used machine learning frameworks
ml_frameworks_counts.head(10).plot(kind='barh', figsize=(10, 6))
plt.title('Top 10 Most Used Machine Learning Frameworks')
plt.xlabel('Number of Respondents')
plt.ylabel('Machine Learning Frameworks')
plt.gca().invert_yaxis()  # Show the highest bar on top
plt.show()


# In[16]:


# Extract the relevant columns for machine learning algorithms (Q18_1 through Q18_14)
ml_algorithm_columns = [f"Q18_{i}" for i in range(1, 15)]  # Adjust range if needed
ml_algorithms_data = df[ml_algorithm_columns]

# Combine all the columns into one series, ignoring NaN values
all_ml_algorithms_used = ml_algorithms_data.stack().reset_index(drop=True)

# Count occurrences of each machine learning algorithm
ml_algorithms_counts = all_ml_algorithms_used.value_counts()

# Plot the top 10 most used machine learning algorithms
ml_algorithms_counts.head(10).plot(kind='barh', figsize=(10, 6))
plt.title('Top 10 Most Used Machine Learning Algorithms')
plt.xlabel('Number of Respondents')
plt.ylabel('Machine Learning Algorithms')
plt.gca().invert_yaxis()  # Show the highest bar on top
plt.show()


# In[30]:


# Extract the relevant columns for computer vision methods (Q19_1 through Q19_8)
computer_vision_columns = [f"Q19_{i}" for i in range(1, 9
                                                    )]  # Adjust range if needed
computer_vision_methods_data = df[computer_vision_columns]

# Combine all the columns into one series, ignoring NaN values
all_computer_vision_methods_used = computer_vision_methods_data.stack().reset_index(drop=True)

# Count occurrences of each computer vision method
computer_vision_methods_counts = all_computer_vision_methods_used.value_counts()

# Plot the top computer vision methods
computer_vision_methods_counts.head(8).plot(kind='barh', figsize=(10, 6))
plt.title('Top Computer Vision Methods Used')
plt.xlabel('Number of Respondents')
plt.ylabel('Computer Vision Methods')
plt.gca().invert_yaxis()  # Show the highest bar on top
plt.show()


# In[32]:


nlp_columns = [f"Q20_{i}" for i in range(1, 7)]  # Adjust range if needed
nlp_methods_data = df[nlp_columns]

# Combine all the columns into one series, ignoring NaN values
all_nlp_methods_used = nlp_methods_data.stack().reset_index(drop=True)

# Count occurrences of each NLP method
nlp_methods_counts = all_nlp_methods_used.value_counts()

# Plot the top NLP methods
nlp_methods_counts.head(6).plot(kind='barh', figsize=(10, 6))
plt.title('Top Natural Language Processing (NLP) Methods Used')
plt.xlabel('Number of Respondents')
plt.ylabel('NLP Methods')
plt.gca().invert_yaxis()  # Show the highest bar on top
plt.show()


# In[33]:


# Extract the relevant columns for pre-trained model weights (Q21_1 through Q21_10)
pretrained_columns = [f"Q21_{i}" for i in range(1, 11)]  # Adjust range if needed
pretrained_weights_data = df[pretrained_columns]

# Combine all the columns into one series, ignoring NaN values
all_pretrained_weights_used = pretrained_weights_data.stack().reset_index(drop=True)

# Count occurrences of each pre-trained model weight
pretrained_weights_counts = all_pretrained_weights_used.value_counts()

# Plot the top pre-trained model weights
pretrained_weights_counts.head(10).plot(kind='barh', figsize=(10, 6))
plt.title('Top Pre-Trained Model Weights Used')
plt.xlabel('Number of Respondents')
plt.ylabel('Pre-Trained Model Weights')
plt.gca().invert_yaxis()  # Show the highest bar on top
plt.show()


# In[27]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Relevant columns
education_column = 'Q8'  # Formal Education
salary_column = 'Q29'    # Salary

# Select and clean data
df_selected = df[[education_column, salary_column]].copy()
df_selected = df_selected.dropna()

# Visualize education vs. salary
plt.figure(figsize=(9, 6))
sns.countplot(data=df_selected, y=education_column, order=df_selected[education_column].value_counts().index)
plt.title('Distribution of Education Levels')
plt.xlabel('Count')
plt.ylabel('Education Level')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_selected, y=education_column, x=salary_column, orient='h')
plt.title('Salary vs. Education Level')
plt.xlabel('Salary')
plt.ylabel('Education Level')
plt.show()

# Correlation analysis (if salary is numeric or encoded)
df_selected['Salary_Numeric'] = pd.to_numeric(df_selected[salary_column], errors='coerce')
df_selected = df_selected.dropna(subset=['Salary_Numeric'])

correlation = df_selected['Salary_Numeric'].corr(df_selected[education_column])
print(f"Correlation between Education Level and Salary: {correlation:.2f}")


# In[3]:


# Display unique values in the salary column to understand the possible answers
unique_salaries = df['Q29'].unique()

# Print the unique values
print("Unique values in the Salary column (Q29):")
print(unique_salaries)


# In[4]:


print("Unique values in the Salary column:")
print(df['Q29'].unique())


# In[6]:


# Test mapping on a subset of the data


# Relevant columns
education_column = 'Q8'  # Formal Education
salary_column = 'Q29'    # Salary
# Map salary ranges to numeric midpoints
def encode_salary_range(salary):
    mapping = {
        '25,000-29,999': 27500,
        '100,000-124,999': 112500,
        '200,000-249,999': 225000,
        '150,000-199,999': 175000,
        '90,000-99,999': 95000,
        '30,000-39,999': 35000,
        '3,000-3,999': 3500,
        '50,000-59,999': 55000,
        '125,000-149,999': 137500,
        '15,000-19,999': 17500,
        '5,000-7,499': 6250,
        '10,000-14,999': 12500,
        '20,000-24,999': 22500,
        '$0-999': 500,
        '7,500-9,999': 8750,
        '4,000-4,999': 4500,
        '80,000-89,999': 85000,
        '2,000-2,999': 2500,
        '250,000-299,999': 275000,
        '1,000-1,999': 1500,
        '$500,000-999,999': 750000,
        '70,000-79,999': 75000,
        '60,000-69,999': 65000,
        '40,000-49,999': 45000,
        '>$1,000,000': 1000000,
        '300,000-499,999': 400000
    }
    return mapping.get(salary, None)



test_mapping = df['Q29'].head(10).apply(encode_salary_range)
print("Mapping test output:")
print(test_mapping)


# In[10]:


import pandas as pd

# Sample DataFrame with categorical columns
data = {
    'JobCode': ['Analyst', 'Scientist', 'Manager'],
    'UsePython': ['Yes', 'No', 'Yes'],
    'UseR': ['No', 'Yes', 'Yes'],
    'Degree': ['Masters', 'PhD', 'Bachelors']
}
df = pd.DataFrame(data)


print(df)

# Ordinal encoding for the 'Degree' column
degree_order = {'Bachelors': 1, 'Masters': 2, 'PhD': 3}
df['Degree'] = df['Degree'].map(degree_order)

# One-hot encoding for the other categorical columns

# Note here that I am putting all of these categorical features into a list
df_encoded = pd.get_dummies(df, columns=['JobCode', 'UsePython', 'UseR'], drop_first=True)

# Display the transformed DataFrame
print


# In[13]:



import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('kaggle_survey_2022_responses.csv', skiprows=1)

# Define the columns for analysis
education_column = 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'
salary_column = 'What is your current yearly compensation (approximate $USD)?'

# Select relevant columns
df_selected = df[[education_column, salary_column]].copy()

# Define ordinal encoding for education levels
education_mapping = {
    'Some college/university study without earning a bachelor’s degree': 1,
    'Bachelor’s degree': 2,
    'Master’s degree': 3,
    'Doctoral degree': 4,
    'I prefer not to answer': 0
}
df_selected[education_column] = df_selected[education_column].map(education_mapping)

# Define mapping for salary ranges to midpoints
salary_mapping = {
    '25,000-29,999': 27500,
    '100,000-124,999': 112500,
    '200,000-249,999': 225000,
    '150,000-199,999': 175000,
    '90,000-99,999': 95000,
    '30,000-39,999': 35000,
    '3,000-3,999': 3500,
    '50,000-59,999': 55000,
    '125,000-149,999': 137500,
    '15,000-19,999': 17500,
    '5,000-7,499': 6250,
    '10,000-14,999': 12500,
    '20,000-24,999': 22500,
    '$0-999': 500,
    '7,500-9,999': 8750,
    '4,000-4,999': 4500,
    '80,000-89,999': 85000,
    '2,000-2,999': 2500,
    '250,000-299,999': 275000,
    '1,000-1,999': 1500,
    '$500,000-999,999': 750000,
    '70,000-79,999': 75000,
    '60,000-69,999': 65000,
    '40,000-49,999': 45000,
    '>$1,000,000': 1000000,
    '300,000-499,999': 400000
}
df_selected[salary_column] = df_selected[salary_column].map(salary_mapping)

# Drop rows with missing values
df_selected.dropna(inplace=True)

# Define features (X) and target (y)
X = df_selected[[education_column]]
y = df_selected[salary_column]

# Encode salary as labels for classification
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_.astype(str)))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importances = rf_model.feature_importances_
plt.figure(figsize=(8, 6))
sns.barplot(x=X.columns, y=feature_importances)
plt.title('Feature Importance in Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.show()

# Decision tree visualization (first tree in the forest)
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], feature_names=X.columns, class_names=label_encoder.classes_, filled=True, fontsize=10)
plt.title('Visualization of a Single Decision Tree in Random Forest')
plt.show()


# In[14]:


from sklearn.tree import plot_tree

# Convert numeric class labels to strings
class_names = [str(cls) for cls in label_encoder.classes_]

# Visualize a single decision tree from the random forest
plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], feature_names=X.columns, class_names=class_names, filled=True, fontsize=10)
plt.title('Visualization of a Single Decision Tree in Random Forest')
plt.show()


# In[15]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Adjust the plot for better readability
plt.figure(figsize=(20, 12))  # Increase figure size
plot_tree(
    rf_model.estimators_[0],  # Use the first tree in the forest
    feature_names=X.columns,  # Feature names (education level)
    class_names=[str(cls) for cls in label_encoder.classes_],  # Class names (salary categories)
    filled=True,  # Fill nodes with colors based on impurity
    rounded=True,  # Round node borders
    fontsize=12,  # Increase font size
    max_depth=3  # Limit depth to 3 for readability
)
plt.title('Improved Visualization of a Decision Tree in Random Forest', fontsize=16)
plt.show()


# In[19]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot a simplified tree
plt.figure(figsize=(20, 12))
plot_tree(
    rf_model.estimators_[0],  # First tree in the forest
    feature_names=['Education Level'],  # Rename feature for clarity
    class_names=['Low', 'Medium', 'High'],  # Rename salary categories
    filled=True,  # Add colors for interpretability
    rounded=True,  # Rounded node shapes
    fontsize=12,  # Increase font size
    max_depth=2  # Limit to first 2 levels
)
# Apply SMOTE to balance the dataset
get_ipython().system('pip install imbalanced-learn')
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)


X_resampled, y_resampled = smote.fit_resample(X, y)
plt.title('Simplified Decision Tree: Education vs Salary', fontsize=16)
plt.show()




# In[11]:


import pandas as pd



# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

# Example: Define ordinal encoding for 'Education' (customize as needed)
ordinal_mappings = {
    'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': {
        'Some college/university study without earning a bachelor’s degree': 1,
        'Bachelor’s degree': 2,
        'Master’s degree': 3,
        'Doctoral degree': 4,
        'I prefer not to answer': 0,
    }
}

# Apply ordinal encoding
for col, mapping in ordinal_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# Apply one-hot encoding for remaining categorical columns
# Exclude already ordinal-encoded columns
remaining_categorical_columns = [col for col in categorical_columns if col not in ordinal_mappings]
df_encoded = pd.get_dummies(df, columns=remaining_categorical_columns, drop_first=True)

# Display the transformed DataFrame
print("Transformed DataFrame Head:")
print(df_encoded.head())

# Save the transformed DataFrame (optional)
df_encoded.to_csv('encoded_kaggle_survey_2022.csv', index=False)


# In[26]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset (replace with your dataset file path)
file_path = 'kaggle_survey_2022_responses.csv'  # Update to your local file path
df = pd.read_csv(file_path, skiprows=1)

# Columns for analysis
education_column = 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'
salary_column = 'What is your current yearly compensation (approximate $USD)?'

# Select relevant columns
df_selected = df[[education_column, salary_column]].copy()

# Map education levels to ordinal values
education_mapping = {
    'Some college/university study without earning a bachelor’s degree': 1,
    'Bachelor’s degree': 2,
    'Master’s degree': 3,
    'Doctoral degree': 4,
    'I prefer not to answer': 0,
    'No formal education past high school': 0
}
df_selected[education_column] = df_selected[education_column].map(education_mapping)

# Map salary ranges to midpoints
salary_mapping = {
    '25,000-29,999': 27500,
    '100,000-124,999': 112500,
    '200,000-249,999': 225000,
    '150,000-199,999': 175000,
    '90,000-99,999': 95000,
    '30,000-39,999': 35000,
    '3,000-3,999': 3500,
    '50,000-59,999': 55000,
    '125,000-149,999': 137500,
    '15,000-19,999': 17500,
    '5,000-7,499': 6250,
    '10,000-14,999': 12500,
    '20,000-24,999': 22500,
    '$0-999': 500,
    '7,500-9,999': 8750,
    '4,000-4,999': 4500,
    '80,000-89,999': 85000,
    '2,000-2,999': 2500,
    '250,000-299,999': 275000,
    '1,000-1,999': 1500,
    '$500,000-999,999': 750000,
    '70,000-79,999': 75000,
    '60,000-69,999': 65000,
    '40,000-49,999': 45000,
    '>$1,000,000': 1000000,
    '300,000-499,999': 400000
}
df_selected[salary_column] = df_selected[salary_column].map(salary_mapping)

# Drop rows with missing values
df_selected.dropna(inplace=True)

# Define features (X) and target (y)
X = df_selected[[education_column]]
y = df_selected[salary_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Feature importance
feature_importances = rf_model.feature_importances_
print("\nFeature Importances:")
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance:.4f}")

# Predict salary for a given education level
sample_education_level = 3  # Example: Master's degree
predicted_salary = rf_model.predict([[sample_education_level]])
print(f"\nPredicted Salary for Education Level {sample_education_level} (Master's degree): ${predicted_salary[0]:,.2f}")


# In[21]:


pip install streamlit


# In[24]:


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path, skiprows=1)
    return df

# Preprocess data
def preprocess_data(df):
    education_column = 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'
    salary_column = 'What is your current yearly compensation (approximate $USD)?'

    # Select relevant columns
    df_selected = df[[education_column, salary_column]].copy()

    # Map education levels to ordinal values
    education_mapping = {
        'Some college/university study without earning a bachelor’s degree': 1,
        'Bachelor’s degree': 2,
        'Master’s degree': 3,
        'Doctoral degree': 4,
        'I prefer not to answer': 0,
        'No formal education past high school': 0
    }
    df_selected[education_column] = df_selected[education_column].map(education_mapping)

    # Map salary ranges to midpoints
    salary_mapping = {
        '25,000-29,999': 27500,
        '100,000-124,999': 112500,
        '200,000-249,999': 225000,
        '150,000-199,999': 175000,
        '90,000-99,999': 95000,
        '30,000-39,999': 35000,
        '3,000-3,999': 3500,
        '50,000-59,999': 55000,
        '125,000-149,999': 137500,
        '15,000-19,999': 17500,
        '5,000-7,499': 6250,
        '10,000-14,999': 12500,
        '20,000-24,999': 22500,
        '$0-999': 500,
        '7,500-9,999': 8750,
        '4,000-4,999': 4500,
        '80,000-89,999': 85000,
        '2,000-2,999': 2500,
        '250,000-299,999': 275000,
        '1,000-1,999': 1500,
        '$500,000-999,999': 750000,
        '70,000-79,999': 75000,
        '60,000-69,999': 65000,
        '40,000-49,999': 45000,
        '>$1,000,000': 1000000,
        '300,000-499,999': 400000
    }
    df_selected[salary_column] = df_selected[salary_column].map(salary_mapping)

    # Drop rows with missing values
    df_selected.dropna(inplace=True)
    
    return df_selected, education_column, salary_column

# Build Streamlit app
def main():
    st.title("Data Scientist Salary Predictor")
    st.write("Predict salaries based on education level.")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload Kaggle Survey Dataset CSV", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)

        # Preprocess data
        df_selected, education_column, salary_column = preprocess_data(df)
        
        # Define features (X) and target (y)
        X = df_selected[[education_column]]
        y = df_selected[salary_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)
        
        # Make predictions and evaluate
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"Model Evaluation:")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R-squared (R2): {r2}")
        
        # Salary Prediction Input
        st.subheader("Predict Salary")
        education_input = st.selectbox(
            "Select Education Level",
            ['No formal education past high school', 
             'Some college/university study without earning a bachelor’s degree',
             'Bachelor’s degree', 
             'Master’s degree', 
             'Doctoral degree']
        )
        
        # Map input to numerical value
        education_level = {
            'No formal education past high school': 0,
            'Some college/university study without earning a bachelor’s degree': 1,
            'Bachelor’s degree': 2,
            'Master’s degree': 3,
            'Doctoral degree': 4
        }.get(education_input, 0)
        
        # Predict salary
        predicted_salary = rf_model.predict([[education_level]])
        st.write(f"Predicted Salary: ${predicted_salary[0]:,.2f}")

if __name__ == "__main__":
    main()

