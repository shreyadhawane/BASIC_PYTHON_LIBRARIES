import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

#1. generate random data with numpy
np.random.seed(0)
branch = np.arange(1, 21)
ENTC_scores = np.random.randint(20, 100, size=20)
CO_scores = np.random.randint(20, 100, size=20)
CIVIL_scores = np.random.randint(20, 100, size=20)


#2.create dataframe with pandas
df = pd.DataFrame({
    'Branch': branch,
    'ENTC':ENTC_scores,
    'CO': CO_scores,
    'CIVIL': CIVIL_scores
})


print("First 5 rows of the DataFrame:")
print(df.head())

#3. basic analysis with pandas
print("\nAverage scores:")
print(df[['ENTC', 'CO', 'CIVIL']].mean())
print("\nStudent with highest Math score:")



#4.visualization with matplotlib
plt.figure(figsize=(8, 5))
plt.plot(df['Branch'],df['ENTC'], marker='o', label='ENTC')
plt.plot(df['Branch'],df['CO'], marker='s', label='CO')
plt.plot(df['Branch'],df['CIVIL'], marker='^', label='CIVIL')
plt.xlabel('Branch')
plt.ylabel('Score')
plt.title('Student Scores by Subject')
plt.legend()
plt.show()

#5.visualization with seaborn
plt.figure(figsize=(8,5))
sns.boxplot(data=df[['ENTC', 'CO', 'CIVIL']])
plt.title('Score Distribution by subject')
plt.ylabel('Score')
plt.show()