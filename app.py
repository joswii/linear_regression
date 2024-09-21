import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# Impoting data preprocessing libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Importing model selection libraries.
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# Importing metrics for model evaluation.
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
# for knn
from sklearn.neighbors import KNeighborsClassifier
# Importing SMOTE for handling class imbalance.
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')


risk_df = pd.read_csv('data_cardiovascular_risk.csv', index_col='id')

numeric_features = []
categorical_features = []

# splitting features into numeric and categoric.

for col in risk_df.columns:  
  if risk_df[col].nunique() > 10:
    numeric_features.append(col) 
  else:
    categorical_features.append(col)

nan_columns = ['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate']

# dropping null values
risk_df.dropna(subset=nan_columns, inplace=True)

risk_df['glucose'] = risk_df.glucose.fillna(risk_df.glucose.median())

# we are going to replace the datapoints with upper and lower bound of all the outliers

def clip_outliers(risk_df):
    for col in risk_df[numeric_features]:
        # using IQR method to define range of upper and lower limit.
        q1 = risk_df[col].quantile(0.25)
        q3 = risk_df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # replacing the outliers with upper and lower bound
        risk_df[col] = risk_df[col].clip(lower_bound, upper_bound)
    return risk_df

risk_df = clip_outliers(risk_df)

risk_df['sex'] = risk_df['sex'].map({'M':1, 'F':0})
risk_df['is_smoking'] = risk_df['is_smoking'].map({'YES':1, 'NO':0})

education_onehot = pd.get_dummies(risk_df['education'], prefix='education')

# drop the original education feature
risk_df.drop('education', axis=1, inplace=True)

# concatenate the one-hot encoded education feature with the rest of the data
risk_df = pd.concat([risk_df, education_onehot], axis=1)


# adding new column PulsePressure
risk_df['pulse_pressure'] = risk_df['sysBP'] - risk_df['diaBP']

# dropping the sysBP and diaBP columns
risk_df.drop(columns=['sysBP', 'diaBP'], inplace=True)

risk_df.drop('is_smoking', axis=1, inplace=True)


X = risk_df.drop('TenYearCHD', axis=1)
y= risk_df['TenYearCHD']


from sklearn.ensemble import ExtraTreesClassifier

# model fitting
model = ExtraTreesClassifier()
model.fit(X,y)

# ranking feature based on importance
ranked_features = pd.Series(model.feature_importances_,index=X.columns)
model_df = risk_df.copy()

X = model_df.drop(columns='TenYearCHD')     # independent features
y = model_df['TenYearCHD']                  # dependent features


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

smote = SMOTE(random_state=33)
# X_train, y_train = smote.fit_resample(X_train, y_train)
# print(X_train.shape)
# print(X_test.shape) 

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)


# Streamlit UI
st.title("Cardiovascular Risk Prediction")

# Input fields for user data
age = st.number_input("Age", min_value=18, max_value=100, value=36)
sex = st.selectbox("Sex", ["Male", "Female"])
cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=100, value=0)
BPMeds = st.selectbox("Blood Pressure Medication", ["No", "Yes"])
prevalentStroke = st.selectbox("Prevalent Stroke", ["No", "Yes"])
prevalentHyp = st.selectbox("Prevalent Hypertension", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
totChol = st.number_input("Total Cholesterol", min_value=100, max_value=500, value=212)
BMI = st.number_input("BMI", min_value=10, max_value=50, value=20)
heartRate = st.number_input("Heart Rate", min_value=40, max_value=150, value=75)
glucose = st.number_input("Glucose", min_value=50, max_value=300, value=75)
education = st.selectbox("Education Level", ["Below 10th", "10th/SSLC", "12th Standard/HSC", "Graduate/Post Graduate"])
pulse_pressure = st.number_input("Pulse Pressure", min_value=20, max_value=100, value=50)

# Convert categorical inputs to numerical
sex = 1 if sex == "Male" else 0
BPMeds = 1 if BPMeds == "Yes" else 0
prevalentStroke = 1 if prevalentStroke == "Yes" else 0
prevalentHyp = 1 if prevalentHyp == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0

# Create a dictionary for education levels
education_dict = {"Below 10th": [1, 0, 0, 0], "10th/SSLC": [1, 1, 0, 0], "12th Standard/HSC": [1, 1, 1, 0], "Graduate/Post Graduate": [1, 1, 1, 1]}

# Create the new data dictionary
new_data = {'age': age,
            'sex': sex,
            'cigsPerDay': cigsPerDay,
            'BPMeds': BPMeds,
            'prevalentStroke': prevalentStroke,
            'prevalentHyp': prevalentHyp,
            'diabetes': diabetes,
            'totChol': totChol,
            'BMI': BMI,
            'heartRate': heartRate,
            'glucose': glucose,
            'education_1.0': education_dict[education][0],
            'education_2.0': education_dict[education][1],
            'education_3.0': education_dict[education][2],
            'education_4.0': education_dict[education][3],
            'pulse_pressure': pulse_pressure}

new_data = pd.DataFrame([new_data])

# Data Scaling
new_data_scaled = scaler.transform(new_data)

# Make the prediction
prediction = knn.predict_proba(new_data_scaled)

# Display the prediction
st.subheader("Prediction:")
st.write("Probability of developing 10-year CHD:", prediction[:, 1][0])
st.write("Risk Assessment:", "High Risk" if prediction[:, 1][0] > 0.5 else "Low Risk") 
