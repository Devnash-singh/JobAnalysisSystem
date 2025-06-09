import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio

# Load data
df = pd.read_csv("job_dataset.csv")

# Clean and preprocess
df['experience'] = df['experience'].fillna(df['experience'].median())
df['salary_range'] = df['salary_range'].fillna("Unknown")
df.dropna(subset=['job_title'], inplace=True)

# Extract min salary
def extract_min_salary(salary_range):
    if isinstance(salary_range, str) and '-' in salary_range:
        try:
            return int(salary_range.split('-')[0].replace('k', '').strip())
        except:
            return np.nan
    return np.nan

df['min_salary_k'] = df['salary_range'].apply(extract_min_salary)

# Extract job category
df['job_category'] = df['job_title'].str.extract(r'(?i)(Engineer|Manager|Analyst|Developer|Intern)', expand=False)

# Remove outliers
q1 = df['min_salary_k'].quantile(0.25)
q3 = df['min_salary_k'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df['min_salary_k'] >= lower_bound) & (df['min_salary_k'] <= upper_bound)]

# Plot 1: Job Category Distribution
job_counts = df['job_category'].value_counts().reset_index()
job_counts.columns = ['job_category', 'count']
fig1 = px.bar(job_counts, x='job_category', y='count',
              title='Job Category Distribution',
              labels={'job_category': 'Job Category', 'count': 'Count'})
pio.write_html(fig1, file="job_category_distribution.html", auto_open=True)

# Plot 2: Salary Boxplot
fig2 = px.box(df, x='job_category', y='min_salary_k',
              title='Salary Range by Job Category',
              labels={'job_category': 'Job Category', 'min_salary_k': 'Min Salary (k)'})
pio.write_html(fig2, file="salary_boxplot.html", auto_open=True)

# Plot 3: Correlation Heatmap
corr = df.select_dtypes(include=[np.number]).corr().round(2)
fig3 = px.imshow(corr, text_auto=True, aspect="auto", 
                 title="Correlation Heatmap of Numeric Features")
pio.write_html(fig3, file="correlation_heatmap.html", auto_open=True)
