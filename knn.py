import pandas as pd
import numpy as np
import math as math
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("wine.data", header=None)

column_names =["Class Labels","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline" ]
df.columns = column_names

df.isnull().sum()
dfm = df.dropna()

x = dfm.iloc[:, 1:].values
y = dfm.iloc[:, 0].values

standard_scaler = StandardScaler()
x = standard_scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=3)

def kNN(x_test,x_train,y_train,distance_metric,k):
    result = [0]*len(x_test)
    if distance_metric == "Euclidean":
        for i2 in range(len(x_test)):
            sum = [0]*len(x_train)
            distance = [0]*len(x_train)
            for i1 in range(len(x_train)):
                for j in range(len(x_train.T)):
                    sum[i1] = sum[i1] + (x_train[i1,j] - x_test[i2,j])**2
                distance[i1] = math.sqrt(sum[i1])
            minclasses = []
            for w in range(k):
                min = np.argmin(distance)
                minclasses.append(y_train[min])
                distance[min] = np.inf
            count = Counter(minclasses)
            minclass = count.most_common(1)[0][0]       
            result[i2] = minclass
    elif distance_metric == "Manhattan":
        for i2 in range(len(x_test)):
            sum = [0]*len(x_train)
            distance = [0]*len(x_train)
            for i1 in range(len(x_train)):
                for j in range(len(x_train.T)):
                    sum[i1] = sum[i1] + abs(x_train[i1,j] - x_test[i2,j])
                distance[i1] = sum[i1]
                minclasses = []
            for w in range(k):
                min = np.argmin(distance)
                minclasses.append(y_train[min])
                distance[min] = np.inf
            count = Counter(minclasses)
            minclass = count.most_common(1)[0][0]       
            result[i2] = minclass
    else:
        raise Exception("Choose a proper distance metric! Type Euclidean or Manhattan in quotation marks as your 4th variable of kNN function.")
    return result

predictions_e = kNN(x_test,x_train,y_train,"Euclidean",11)
print("          Euclidean Results\n\n    Accuracy: ",accuracy_score(predictions_e,y_test))
print("\n    Classification report:\n\n",classification_report(predictions_e,y_test))
print("\n    Confusion matrix:\n",confusion_matrix(predictions_e,y_test))

predictions_m = kNN(x_test,x_train,y_train,"Manhattan",11)
print("          Manhattan Results\n\n    Accuracy: ",accuracy_score(predictions_m,y_test))
print("\n    Classification Report:\n\n",classification_report(predictions_m,y_test))
print("\n    Confusion Matrix:\n",confusion_matrix(predictions_m,y_test))