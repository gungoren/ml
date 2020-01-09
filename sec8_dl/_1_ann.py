#
#
# Mehmet Güngören (mehmetgungoren@lyrebirdstudio.net)
#
#
# 1. libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# 2. preprocessing
# 2.1. import data
df = pd.read_csv("Churn_Modelling.csv")

X = df.iloc[:, 3:-1]
y = df.iloc[:, -1:]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X.iloc[:,1] = le.fit_transform(X.iloc[:, 1])

le2 = LabelEncoder()
X.iloc[:,2] = le2.fit_transform(X.iloc[:,2])

ohe = OneHotEncoder(categories='auto')
geography = ohe.fit_transform(X[['Geography']]).toarray()

geographyDf = pd.DataFrame(data=geography, columns=['fr', 'es', 'de'], index=range(len(geography)))
X = pd.concat([X, geographyDf], axis=1)
X = X.drop("Geography", axis=1)

# 3. split test and train data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# 4. attribute scaling
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=12))
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=50)
y_pred = classifier.predict(X_test)

y_pred = (y_pred >= 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)







