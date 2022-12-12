from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
#Import dataset form tensorflow
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
X_train
len(X_train)
len(X_test)
plt.matshow(X_train[0])
#Standarize the data scale
X_train_scaled = X_train/255
X_test_scaled = X_test/255
X_train_scaled.shape
X_test_scaled.shape
#Image data is in three dimensional, convert it to two dimensional
X_train_scaled = X_train_scaled.reshape(len(X_train),28*28)
X_test_scaled = X_test_scaled.reshape(len(X_test),28*28)
#Training the data using Logistic Regression model
model.fit(X_train_scaled,y_train)
model.score(X_test_scaled,y_test)
#92.58% score
y_predicted = model.predict(X_test_scaled)
#Check the predicted value with the image
y_predicted[246]
plt.matshow(X_test[246])
#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm
import seaborn as sn
plt.figure(figsize=(15,15))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")

#In Logistic regression the model stopped, solver = lgfgs could not converge. 
#To reach the local optima keras neural network approach can be used
modelk = keras.Sequential([keras.layers.Dense(10, input_shape = (784,), activation = 'sigmoid')])
modelk.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
modelk.fit(X_train_scaled, y_train,epochs=5)
modelk.evaluate(X_test_scaled,y_test)
#Accuracy is similar to LogisticRegression model
#Using hidden layer
modelkh = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

modelkh.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

modelkh.fit(X_train_scaled, y_train, epochs=5)
#Hidden layer increases accuracy to 97.42%
#Using flatten layer to avoid reshaping the data
modelkf = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

modelkf.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)
#This gives accuracy of 93.91%










