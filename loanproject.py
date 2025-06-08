import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Define folder to save images
desktop_folder = r"C:\Users\eq5cd\OneDrive\Desktop\loanproject"
if not os.path.exists(desktop_folder):
    os.makedirs(desktop_folder)

df = pd.read_csv("C:/Users/eq5cd/Downloads/Training Dataset.csv")
print(df.head())
print(df.info())

print(df.isnull().sum())
# Fill missing categorical columns with mode
categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']

for col in categorical_cols:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)

# Fill missing numerical column LoanAmount with median
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Verify missing values are handled
print(df.isnull().sum())
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# Columns to encode
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print(df.head())

# Features and Target
X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area']]

y = df['Loan_Status']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize model
model = LogisticRegression(max_iter=1000)

# Fit on training data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# New applicant data as a 2D array
new_applicant = np.array([[1, 1, 0, 0, 0, 5000, 1500, 150, 360, 1, 2]])

# Predict probability
probability = model.predict_proba(new_applicant)
prob = model.predict(new_applicant)

print("Probability of rejection (0):", probability[0][0])
print("Probability of approval (1):", probability[0][1])
print(prob)

# Plot and save Loan Status Distribution
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Approval Distribution")
plt.savefig(os.path.join(desktop_folder, 'loan_status_distribution.png'))
plt.show()

# Map encoded Property_Area numbers to readable labels
area_mapping = {0: 'Rural', 1: 'Semiurban', 2: 'Urban'}
df['Property_Area_Label'] = df['Property_Area'].map(area_mapping)

# Plot and save Loan Approval by Property Area
sns.countplot(x='Property_Area_Label', hue='Loan_Status', data=df)
plt.title("Loan Approval by Property Area")
plt.savefig(os.path.join(desktop_folder, 'loan_approval_by_property_area.png'))
plt.show()

# Plot and save Boxplot for Applicant Income by Loan Status
sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df)
plt.title("Applicant Income by Loan Status")
plt.savefig(os.path.join(desktop_folder, 'applicant_income_by_loan_status.png'))
plt.show()

# Plot and save Correlation Heatmap
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Loan Features")
plt.savefig(os.path.join(desktop_folder, 'correlation_heatmap.png'))
plt.show()

# Plot and save Distribution of Loan Amount
plt.figure(figsize=(8, 5))
sns.histplot(df['LoanAmount'], kde=True, color='purple')
plt.title("Distribution of Loan Amount")
plt.xlabel("Loan Amount (in thousands)")
plt.ylabel("Number of Applicants")
plt.savefig(os.path.join(desktop_folder, 'distribution_loan_amount.png'))
plt.show()
