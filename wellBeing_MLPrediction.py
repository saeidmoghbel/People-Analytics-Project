import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
sns.set()

df = pd.read_csv("/Users/saeidmoghbel/Downloads/PA_ProjectWorkII/employee_wellbeing.csv", sep=',')

print(df.columns)

# preprocessing and data cleaning
'''df = df.drop(columns=['STATUS', 'PERSONAL_AWARDS', 'HEALTHY_DIET', 'LOST_VACATION', 'SLEEP_HOURS', 'DAILY_STEPS',
                      'FLOW', 'ACHIEVEMENT', 'SOCIAL_NETWORK', 'SUPPORTING_OTHERS'])'''

categorical_df = ['AGE', 'GENDER', 'STATUS', 'EMPLOYMENT', 'SALARY']
mode_df = df[categorical_df].mode().iloc[0]
categories = df.loc[:, categorical_df] = df[categorical_df].fillna(mode_df)
print(categories.isnull().sum())

print(categories.head())

numerical_df = ['SUFFICIENT_INCOME', 'TO_DO_COMPLETED',	'DAILY_STRESS',	'CORE_CIRCLE',	'SUPPORTING_OTHERS',	'SOCIAL_NETWORK',	'ACHIEVEMENT',	'FLOW',	'DAILY_STEPS',	'SLEEP_HOURS',	'LOST_VACATION',	'PERSONAL_AWARDS',	'TIME_FOR_HOBBY',	'HEALTHY_DIET',	'WORK_LIFE_BALANCE_SCORE']
mode_df = df[numerical_df].mode().iloc[0]
df.loc[:, numerical_df] = df[numerical_df].fillna(mode_df)
print(df[numerical_df].isnull().sum())

'''label_encoder = LabelEncoder()
label_encoder.fit(categories)
encoded_labels = label_encoder.transform(categories)
print(encoded_labels)'''
encoded_df = pd.get_dummies(df[categorical_df])
print(encoded_df.head())

#df = df.fillna(df.mean())
#df.isnull().values.any()
df = df.dropna(axis=0, how='any')
df.isnull().values.any()

print(df.info())
print(df.head())

# first glance at our dataset
print(df.describe())


#df['AGE'] = df['AGE'].astype(int)
#df['AGE'] = df['AGE'].apply(lambda x: x[0] if isinstance(x, list) else x)

#df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce').astype('Int64')
# Flatten the 'AGE' column
#df['AGE'] = df['AGE'].apply(lambda x: x[0] if isinstance(x, list) else x)

# Convert 'AGE' column to integer
#df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce').astype('Int6ß4')


df = df.drop(['AGE'], axis=1)
df['GENDER'] = df['GENDER'].map({'Male': 0, 'Female': 1})
df['STATUS'] = df['STATUS'].map({'single': 0, 'married': 1, 'divorced': 2, 'in a relation': 3})
df['EMPLOYMENT'] = df['EMPLOYMENT'].map({'flight_attendant': 0, 'checkin_agent': 1})
df['SALARY'] = df['SALARY'].map({'Low': 0, 'Medium': 1, 'High': 2})
#df['AGE'] = df['AGE'].map({'Less than 25': [25], '25 to 35': [35], '36 to 50': [50], '51 or more': [60]})

# Confirm if NaN values are removed
X = df.drop(['WORK_LIFE_BALANCE_SCORE'], axis=1)
y = df['WORK_LIFE_BALANCE_SCORE']

print(df.head())
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
#model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
predicted_value = model.predict([[1,1, 1, 1, 1, 6, 2, 5, 0, 5, 2, 4, 5, 7, 5, 4, 0, 3, 609.5]])
actual_value = df.loc[1, 'WORK_LIFE_BALANCE_SCORE']
print(predicted_value, actual_value)
#input_array = np.array([list(predicted_value.values())])
#comparison_df = pd.DataFrame({'Real Value': y_test, 'Predicted Value': y_pred})
#print(comparison_df)

# Present the difference between real and predicted values
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

plt.figure(figsize=(15,10))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.show()

pred_y_df = pd.DataFrame({'Actual Value': y_test, 'Predicted value': y_pred, 'Difference': y_test - y_pred})



# Reshape the predicted value into a 2D array
predicted_value = model.predict(predicted_value.reshape(-1, 1))

# Print the predicted WLB score for the new employee
print("Predicted WLB score for the new employee:", predicted_value)