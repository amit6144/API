import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error
import pickle

df=pd.read_csv('credit_train.csv')
print(df.head())
X=df[['Loan Status','Current Loan Amount','Annual Income','Current Credit Balance']]
X.fillna(X.mean(),inplace=True)
Y=df[['Credit Score']]
Y.fillna(Y.mean(),inplace=True)
LE=LabelEncoder()
X['Loan Status']=LE.fit_transform(X['Loan Status'])
X=np.array(X)
Y=np.array(Y)
trainx,testx,trainy,testy=tts(X,Y,test_size=0.3,random_state=30)

model=LinearRegression()
model.fit(trainx,trainy)
ypred=model.predict(testx)
error=mean_squared_error(testy,ypred)
print(' Error',error)

pickle.dump(model,open('CreditScore.pkl','wb'))
test=np.array([1,215908.0,633992.0,82859.0]).reshape(1,-1)
res=model.predict(test)
ress=res[0][0]
print(ress)