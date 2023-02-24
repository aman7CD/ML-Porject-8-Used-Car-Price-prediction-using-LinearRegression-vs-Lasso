#Importing the Dependencies
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



# Data Colection and Preprocessing

data = pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/car data.csv")

data.isnull().sum()

data.head()

data.drop(["Car_Name"], axis=1, inplace=True)



# Converting data from text to numerical

data.replace(to_replace=["Petrol","Diesel","CNG","Manual","Automatic","Dealer","Individual"], value=[0,1,2,0,1,0,1,], inplace=True)

data.head()

# Splitting the Data

x = data.drop(["Selling_Price"],axis=1)

y = data["Selling_Price"]

xtn,xtt,ytn,ytt = train_test_split(x,y, test_size=0.1, random_state=2, )



# Training The Model

model = LinearRegression()

model.fit(xtn,ytn)

model1 =Lasso()

model1.fit(xtn,ytn)



# Model Evaluation Through r2 Score and MSE

y_pred = model.predict(xtt)

r2score = r2_score(ytt,y_pred)

r2score


y_pred1 = model1.predict(xtt)

r2score1 = r2_score(ytt,y_pred1)

r2score1

print(f"The r2score of linear regression is {r2score} and the r2score of the lasso is {r2score1}")

rmse= mean_squared_error(ytt,y_pred, squared=False)

rmse1= mean_squared_error(ytt,y_pred1, squared=False)

print(f"The RMSE of linear regression is {rmse} and the r2score of the lasso is {rmse1}")
