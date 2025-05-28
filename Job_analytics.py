import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('job.csv')  # Adjust path if needed

# View the first few rows
print(df.head())

# Basic info
print(df.info())

# Check missing values
print(df.isnull().sum())

# Fill missing values
df['location'].fillna('Unknown', inplace=True)
df['salary'].fillna(df['salary'].median(), inplace=True)

# Drop rows where job_title is missing
df.dropna(subset=['job_title'], inplace=True)

# Create a simple job type from title
df['job_type'] = df['job_title'].apply(lambda x: x.split()[0])

# Extract year from posting date
df['posting_year'] = pd.to_datetime(df['posting_date'], errors='coerce').dt.year

# Strip whitespace and fix casing
df['company'] = df['company'].str.strip().str.title()
df['location'] = df['location'].replace({'Newyork': 'New York', 'San Fransisco': 'San Francisco'})

Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1

# Remove outliers
df = df[(df['salary'] >= Q1 - 1.5 * IQR) & (df['salary'] <= Q3 + 1.5 * IQR)]

print(df.describe())
print(df['job_type'].value_counts())
print(df['location'].value_counts())



# Jobs per year
df.groupby('posting_year').size().plot(kind='bar', title='Jobs per Year')
plt.show()

# Average salary per job type
df.groupby('job_type')['salary'].mean().sort_values().plot(kind='barh', title='Average Salary by Job Type')
plt.show()

