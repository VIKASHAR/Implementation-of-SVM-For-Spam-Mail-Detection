# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas. 
3. Get the information regarding the null values present in the dataframe.
4. Split the data into training and testing sets.
5. Convert the text data into a numerical representation using CountVectorizer.
6. Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7. Finally, evaluate the accuracy of the model.

## Program:
```python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Vikash A R
RegisterNumber:  212222040179
*/
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
```

## Output:

## Result Output

![282257583-78ccb346-ca7c-4a33-ad4c-e3355e1fddc6](https://github.com/VIKASHAR/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119405655/2006ebd7-0da9-4d49-b536-6e8ae84a09e3)

## data.head()

![282257589-139f19db-04ee-4e44-b04a-5f231988b90b](https://github.com/VIKASHAR/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119405655/332aee83-052e-43df-85c5-5a5892d0a8b0)

## data.info()

![282257595-646ac557-8f21-442c-8783-6a1085ec89fd](https://github.com/VIKASHAR/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119405655/2b54c627-0df8-49aa-a714-6a63ebb24678)

## data.isnull().sum()

![282257608-2eba109c-0bdd-468a-8bcf-d9258c23f8ef](https://github.com/VIKASHAR/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119405655/ca6f7181-50ea-4907-9b9f-dcb61ef6ca9f)

![282257631-b0e5dbc6-7c5b-40fe-b610-86fc97828918](https://github.com/VIKASHAR/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119405655/a1abaad9-87b1-400e-8414-b52a0cc7985b)


## Y_prediction Value

![282257837-6a911f5c-1e40-4047-9371-07a94f012cef](https://github.com/VIKASHAR/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119405655/ebcbd919-d4b3-4fb9-96a2-ee33be6a8a0b)

## Accuracy Value

![282257853-bad89364-aef2-4652-806d-09c5760c041e](https://github.com/VIKASHAR/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119405655/22e30b8b-47e1-4c00-a8e3-6d3b711f641f)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
