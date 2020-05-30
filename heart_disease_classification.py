import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.show()

df = pd.read_csv('heartdf.csv')

X = df.drop('target',axis=1).values
y = df['target'].values

#Splitting the data into train and test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=14)

#Scaling the data for the model

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Creating the model for the ANN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

model = Sequential()

model.add(Dense(15,input_dim=13,activation='relu')) #INPUT LATER
model.add(Dropout(0.5))

model.add(Dense(12,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(12,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid')) #OUTPUT LAYER

model.compile(optimizer='adam',
              loss='binary_crossentropy')

#Creating a callback for ANN model
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=5)

#Fitting the ANN model with the training dataset

model.fit(x=X_train,y=y_train,
          epochs=600,validation_data=[X_test,y_test],
          callbacks=[early_stopping])

#AFTER running model will be fit. Now we will make predictions on the test dataset using our model and will then compare 
predictions = model.predict_classes(X_test)

#Evaluating our model
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

