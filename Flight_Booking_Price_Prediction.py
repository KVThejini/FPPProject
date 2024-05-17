import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing data

df = pd.read_excel("D:\Datasets\FlightBooking.xlsx")
df.head()

#Loading Data

df.shape
df.info()
df.describe()

#Checking missing values

df.isnull().sum()
mode_route = df['Route'].mode()[0]
df['Route'] = df['Route'].fillna(mode_route)
mode_stops = df['Total_Stops'].mode()[0]
df['Total_Stops'] = df['Total_Stops'].fillna(mode_stops)
print(df.isnull().sum())

#Data visualization

plt.figure(figsize=(15,5))
sns.lineplot(x=df['Airline'],y=df['Price'])
plt.title("Airline vs Price")
plt.xlabel("Airline",fontsize = 15)
plt.ylabel("Price",fontsize = 15)
plt.show()

from datetime import datetime

# Assuming your DataFrame is named 'df'
# Convert 'Date_of_Journey' to datetime format
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
# Calculate the difference between the current date and 'Date_of_Journey'
current_date = pd.to_datetime('now').normalize()  # Get current date without time
df['Days_to_Departure'] = (current_date - df['Date_of_Journey']).dt.days
# Display the DataFrame with the new column
print(df)

plt.figure(figsize=(15,5))
sns.lineplot(data = df,x='Days_to_Departure',y='Price',color='blue')
plt.title("Days left for departure Vs Ticket Price")
plt.xlabel("Days left for Departure",fontsize = 15)
plt.ylabel("Ticket Price",fontsize = 15)
plt.show()

plt.figure(figsize=(15,5))
sns.barplot(x=df['Airline'],y=df['Price'])
plt.title("Airline vs Price")
plt.xlabel("Airline",fontsize = 15)
plt.ylabel("Price",fontsize = 15)
plt.show()

fig,ax = plt.subplots(1,2,figsize=(20,6))
sns.lineplot(x='Days_to_Departure',y='Price',data=df,hue='Source',ax=ax[0])
sns.lineplot(x='Days_to_Departure',y='Price',data=df,hue='Destination',ax=ax[1])
plt.title("Days left for departure Vs Ticket Price")
plt.xlabel("Days left for Departure",fontsize = 15)
plt.ylabel("Ticket Price",fontsize = 15)
plt.show()

#Performing One Hot Encoding for categorical features of a dataframe

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Airline'] = le.fit_transform(df['Airline'])
df['Date_of_Journey'] = le.fit_transform(df['Date_of_Journey'])
df['Source'] = le.fit_transform(df['Source'])
df['Destination'] = le.fit_transform(df['Destination'])
df['Route'] = le.fit_transform(df['Route'])
df['Dep_Time'] = le.fit_transform(df['Dep_Time'])
df['Arrival_Time'] = le.fit_transform(df['Arrival_Time'])
df['Duration'] = le.fit_transform(df['Duration'])
df['Total_Stops'] = le.fit_transform(df['Total_Stops'])
df['Additional_Info'] = le.fit_transform(df['Additional_Info'])
df.info()

#feature selection
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.show()

#vif
from statsmodels.stats.outliers_influence import variance_inflation_factor
col_list =[]
for col in df.columns:
    if((df[col].dtype !='object') & (col!='Price')):
        col_list.append(col)
x = df[col_list]
vif_data = pd.DataFrame()
vif_data['feature']= x.columns
vif_data['VIF']=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]
print(vif_data)

#droping columns having vif less than 6
df = df.drop(columns=['Route','Arrival_Time','Additional_Info','Days_to_Departure'])

from statsmodels.stats.outliers_influence import variance_inflation_factor
col_list =[]
for col in df.columns:
    if((df[col].dtype !='object') & (col!='Price')):
        col_list.append(col)
x = df[col_list]
vif_data = pd.DataFrame()
vif_data['feature']= x.columns
vif_data['VIF']=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]
print(vif_data)

#Linear Regression
X = df.drop(columns=['Price'])
y= df['Price']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
diff = pd.DataFrame(np.c_[y_test,y_pred],columns=['Actual_Value','Predicted_Value'])
print(diff)

#Calculating R2,MAE,MAPE,MSE,RMSE for linear regression
from sklearn.metrics import r2_score
r2score = r2_score(y_test,y_pred)
print("R2 Score in LR Model:",r2score)
from sklearn import metrics
mean_abs_error = metrics.mean_absolute_error(y_test,y_pred)
print("Mean Abosulte error in LR Model:",mean_abs_error)
from sklearn.metrics import mean_absolute_percentage_error
mean_abs_per_error = mean_absolute_percentage_error(y_test,y_pred)
print("Mean Absolute percentage error in LR Model:",mean_abs_per_error)
mean_sq_error = metrics.mean_squared_error(y_test,y_pred)
print("Mean square error in LR Model:",mean_sq_error)
root_mean_sq_error = np.sqrt(mean_sq_error)
print("Root_Mean_Square_Error in LR Model:",root_mean_sq_error)

#Linear Regression graph
sns.distplot(y_test,label='Actual value')
sns.distplot(y_pred,label='predicted value')
plt.legend()

#Desicion Tree
X1 = df.drop(columns=['Price'])
y1= df['Price']
from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test = train_test_split(X1,y1,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x1_train=sc.fit_transform(x1_train)
x1_test=sc.transform(x1_test)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x1_train,y1_train)
y1_pred=dt.predict(x1_test)
diff1 = pd.DataFrame(np.c_[y1_test,y1_pred],columns=['Actual_Value','Predicted_Value'])
print(diff1)

#Calculating R2,MAE,MAPE,MSE,RMSE for Decision Tree Model
from sklearn.metrics import r2_score
r2score = r2_score(y1_test,y1_pred)
print("R2 Score in DT Model:",r2score)
from sklearn import metrics
mean_abs_error = metrics.mean_absolute_error(y1_test,y1_pred)
print("Mean Abosulte error in DT Model :",mean_abs_error)
from sklearn.metrics import mean_absolute_percentage_error
mean_abs_per_error = mean_absolute_percentage_error(y1_test,y1_pred)
print("Mean Absolute percentage error in DT Model :",mean_abs_per_error)
mean_sq_error = metrics.mean_squared_error(y1_test,y1_pred)
print("Mean square error in DT Model:",mean_sq_error)
root_mean_sq_error = np.sqrt(mean_sq_error)
print("Root_Mean_Square_Error in DT Model:",root_mean_sq_error)

#Descision Tree graph
sns.distplot(y1_test,label='Actual value')
sns.distplot(y1_pred,label='predicted value')
plt.legend()

#Random Forest 
X2 = df.drop(columns=['Price'])
y2= df['Price']
from sklearn.model_selection import train_test_split
x2_train,x2_test,y2_train,y2_test = train_test_split(X2,y2,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x2_train=sc.fit_transform(x2_train)
x2_test=sc.transform(x2_test)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x2_train,y2_train)
y2_pred=rf.predict(x2_test)
diff2 = pd.DataFrame(np.c_[y2_test,y2_pred],columns=['Actual_Value','Predicted_Value'])
print(diff2)

#Calculating R2,MAE,MAPE,MSE,RMSE for Random Forest Model
from sklearn.metrics import r2_score
r2score = r2_score(y1_test,y1_pred)
print("R2 Score in RF Model:",r2score)
from sklearn import metrics
mean_abs_error = metrics.mean_absolute_error(y2_test,y2_pred)
print("Mean Abosulte error in RF Model :",mean_abs_error)
from sklearn.metrics import mean_absolute_percentage_error
mean_abs_per_error = mean_absolute_percentage_error(y2_test,y2_pred)
print("Mean Absolute percentage error in RF Model :",mean_abs_per_error)
mean_sq_error = metrics.mean_squared_error(y2_test,y2_pred)
print("Mean square error in RF Model:",mean_sq_error)
root_mean_sq_error = np.sqrt(mean_sq_error)
print("Root_Mean_Square_Error in RF Model:",root_mean_sq_error)

#Random Forest graph
sns.distplot(y2_test,label='Actual value')
sns.distplot(y2_pred,label='predicted value')
plt.legend()
