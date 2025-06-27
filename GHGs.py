import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

excel_file = r"C:\Users\Hanu\Downloads\af60b10b8dad38110304.xlsx"
years = range(2010, 2017)

all_data = []
for year in years:
    try:
        df_com = pd.read_excel(excel_file, sheet_name=f'{year}_Detail_Commodity')
        df_ind = pd.read_excel(excel_file, sheet_name=f'{year}_Detail_Industry')
        df_com["Source"] = "Commodity"
        df_ind["Source"] = "Industry"
        df_com["Year"] = df_ind["Year"] = year
        df_com.columns = df_com.columns.str.strip()
        df_ind.columns = df_ind.columns.str.strip()
        df_com.rename(columns={'Commodity Code': 'Code', 'Commodity Name': 'Name'}, inplace=True)
        df_ind.rename(columns={'Industry Code': 'Code', 'Industry Name': 'Name'}, inplace=True)
        combined = pd.concat([df_com, df_ind], ignore_index=True)
        all_data.append(combined)
    except Exception as e:
        print(f"Error processing year {year}: {e}")

df = pd.concat(all_data, ignore_index=True)
len(df)

print("\nColumns in Final Combined DF:\n", df.columns)
missing_values = df.isnull().sum()
print("\nMissing Values in Each Column:\n", missing_values)
#week2 from here on ----->

# modification 1 dropping empty column
df.drop(columns=['Unnamed: 7'], inplace=True)

# modification 2 Removing duplicate rows
df.drop_duplicates(inplace=True)

# 3 Fill all missing values with mode
for column in df.columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\nColumns in Final Combined DF:\n", df.columns)
print("\nMissing Values:\n", df.isnull().sum())
sns.histplot(df['Supply Chain Emission Factors with Margins'], bins=50, kde=True)
plt.title('Target Variable Distribution')
plt.show()
print(df['Substance'].value_counts())
print(df['Unit'].value_counts())
print(df['Source'].value_counts())

df['Substance'] = df['Substance'].map({'carbon dioxide':0, 'methane':1, 'nitrous oxide':2, 'other GHGs':3})
df['Unit'] = df['Unit'].map({'kg/2018 USD, purchaser price':0, 'kg CO2e/2018 USD, purchaser price':1})
df['Source'] = df['Source'].map({'Commodity':0, 'Industry':1})

df.drop(columns=['Name','Code','Year'], inplace=True)

# Define features and target
X = df.drop(columns=['Supply Chain Emission Factors with Margins'])
y = df['Supply Chain Emission Factors with Margins']    

#univariate analysis with piechart
source_counts = df['Source'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(
    source_counts, 
    labels=['Commodity', 'Industry'], 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=["#a0ff99",'#66b3ff']
)
plt.title('Source Proportion (Commodity vs Industry)')
plt.axis('equal')
plt.show()

#multivariate analysis with bubble chart
plt.figure(figsize=(12, 7))
plt.scatter(
    x=df['Supply Chain Emission Factors without Margins'],
    y=df['Margins of Supply Chain Emission Factors'],
    s=df['Supply Chain Emission Factors with Margins'] * 5,  # Scale size of bubbles
    c=df['Substance'],  # Color by substance category
    cmap='viridis',
    alpha=0.6,
    edgecolors='w',
    linewidth=0.5
)

plt.xlabel('Emission Factors without Margins')
plt.ylabel('Margins of Emission Factors')
plt.title('Bubble Plot: Emission vs Margin (Size = With Margins, Color = Substance)')
plt.colorbar(label='Substance Type (encoded)')
plt.grid(True)
plt.tight_layout()
plt.show()

