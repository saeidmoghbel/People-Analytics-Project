import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
sns.set()

df = pd.read_csv("/Users/saeidmoghbel/Downloads/PA_ProjectWorkII/employee_wellbeing.csv", sep=',')

print(df.columns)

# preprocessing and data cleaning
#df = df.drop(columns=['STATUS', 'PERSONAL_AWARDS', 'HEALTHY_DIET', 'LOST_VACATION', 'SLEEP_HOURS', 'DAILY_STEPS',
 #                     'FLOW', 'ACHIEVEMENT', 'SOCIAL_NETWORK', 'SUPPORTING_OTHERS'])
df.isnull().values.any()
df = df.dropna(axis=0, how='any')
df.isnull().values.any()

print(df.info())
print(df.head())

# first glance at our dataset
print(df.describe())


sum_daily_stress_age = df.groupby('AGE')['DAILY_STRESS'].sum()

# Plot the barplot
plt.figure(figsize=(20, 10))
sns.barplot(x=sum_daily_stress_age.index, y=sum_daily_stress_age.values)
plt.title('Total Daily Stress by Age Group')
plt.ylabel('Total Daily Stress')
plt.xlabel('Age Group')
plt.gca().set_xticks([0, 1, 2, 3])
plt.gca().set_xticklabels(['Less than 25', '25 to 35', '36 to 50', '51 or more'], rotation=45, ha='right')
plt.show()

print(sum_daily_stress_age.index)


sum_daily_stress_employment = df.groupby('EMPLOYMENT')['DAILY_STRESS'].sum()

# Create the bar plot
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
sns.barplot(x=sum_daily_stress_employment.index, y=sum_daily_stress_employment.values)
plt.title('Total Daily Stress by Employment')
plt.ylabel('Total Daily Stress')
plt.xlabel('Employment')
plt.xticks(ticks=[0,1], labels=['flight_attendant', 'checkin_agent'], rotation=45, ha='right')
plt.show()


sum_time_for_hobby = df.groupby('GENDER')['TIME_FOR_HOBBY'].sum()


# Plotting
plt.figure(figsize=(20, 10))
ax = sns.barplot(x=['male', 'female'], y=sum_time_for_hobby.values)
plt.title('Total number of hobbies for each gender')
plt.xlabel('Gender')
plt.ylabel('Total Time for Hobby')

# Getting the x-axis tick labels
x_labels = ax.get_xticklabels()
# Setting the x-axis tick labels
ax.set_xticklabels(x_labels)
ax.set_xticks(['male', 'female'])

plt.show()


#non_numeric_columns = df.select_dtypes(exclude=['number']).columns
#print("Non-numeric columns:", non_numeric_columns)


columns_to_convert = ['AGE', 'GENDER', 'STATUS', 'EMPLOYMENT', 'SALARY']
df.columns = df.columns.str.strip()
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')


columns_for_correlation = ['AGE',	'GENDER',	'STATUS',	'EMPLOYMENT',	'SUFFICIENT_INCOME',	'SALARY',	'TO_DO_COMPLETED',	'DAILY_STRESS',
                           'CORE_CIRCLE',	'SUPPORTING_OTHERS',	'SOCIAL_NETWORK',	'ACHIEVEMENT',	'FLOW',	'DAILY_STEPS',	'SLEEP_HOURS',	
                           'LOST_VACATION',	'PERSONAL_AWARDS',	'TIME_FOR_HOBBY',	'HEALTHY_DIET',	'WORK_LIFE_BALANCE_SCORE']
correlation_matrix = df[columns_for_correlation].corr()

plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='vlag', fmt="0.3f")
plt.title('Correlation Heatmap of Work-Life Balance Score and Other Attributes')
plt.show()
print(correlation_matrix)
correlation_matrix = correlation_matrix.unstack()
correlation_matrix = [(correlation_matrix) > 0.5]