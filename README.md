# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and.duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. Apply new unknown values
 

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Akshaikhanna D
RegisterNumber:212223040010


import pandas as pd
df=pd.read_csv("Placement_Data.csv")
print(df.head())

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
print(df1.head())

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
print(df1)

x=df1.iloc[:,:-1]
print(x)

y=df1["status"]
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
## Original Data:

<img width="1066" height="206" alt="image" src="https://github.com/user-attachments/assets/20f9d379-0b1e-484b-a507-fbefecc3d492" />



## After Removing:
<img width="832" height="206" alt="image" src="https://github.com/user-attachments/assets/3e879643-8fd6-47db-a5d6-e8413d4f0332" />


## Null Data:
<img width="564" height="432" alt="image" src="https://github.com/user-attachments/assets/cd0f8286-2e7c-412b-8768-8028147c7f85" />



## Label Encoder:
<img width="970" height="417" alt="image" src="https://github.com/user-attachments/assets/e7f6bdee-d7d6-4477-8821-13e30fef50b8" />



## x:
<img width="471" height="377" alt="image" src="https://github.com/user-attachments/assets/73f3618f-c81f-4342-85c4-29658e61198c" />



## y:
<img width="505" height="334" alt="image" src="https://github.com/user-attachments/assets/1b6c7c55-684a-45af-a750-2511a1aa1677" />


## Y_Prediction:
<img width="927" height="82" alt="image" src="https://github.com/user-attachments/assets/7a2772e1-b644-4d8b-8624-2595523e1d7d" />

## Accuracy:
<img width="227" height="43" alt="image" src="https://github.com/user-attachments/assets/26ff6a84-2792-4a40-bc63-f19840b2304c" />

## Cofusion:
<img width="137" height="90" alt="image" src="https://github.com/user-attachments/assets/8a7dacf2-a69e-4775-b651-e2609c52e98b" />


## Classification:


<img width="1050" height="306" alt="image" src="https://github.com/user-attachments/assets/7c568729-4cf2-4404-b8bc-1f01f7b9d35d" />







## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
