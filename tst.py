import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

#checking the labels of the datasets
labels = pd.read_csv("labels.csv")
# print(labels.describe())
# print(labels.head())

#print(labels.head())
# print(labels["breed"].value_counts())

#print(labels["breed"].value_counts().mean())

# pathname from the dataset
filenames=["train/train/" +fname+".jpg" for fname in labels["id"]]
#print(filenames[:5])

labels_np = labels["breed"].to_numpy()
#print(labels_np)
#print(len(labels_np))


# find unique label values
unique_breeds = np.unique(labels_np)
#print(unique_breeds)
# print(labels.breed.nunique())
# print(labels.isnull().sum())

H=128
W=128
C=3

train_file_location = 'train/train/' 
train_data =labels.assign(img_path = lambda x : train_file_location + x['id'] + '.jpg')
#print(train_data.head())

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img,img_to_array
X = np.array([img_to_array(load_img(img,target_size = (H,W))) for img in train_data['img_path'].values.tolist()])
#print(X.shape)
Y = pd.get_dummies(train_data['breed'])
#print(Y.shape)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test  = train_test_split(X,Y,test_size = 0.25)
#print(X_train.shape,Y_train.shape)
#print(X_test.shape,Y_test.shape)


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,Activation


model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(H,W,C)))
model.add(Activation('relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))

model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))


model.add(Dense(Y.shape[1]))
model.add(Activation('softmax'))

#model.summary()
batch=32

model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['accuracy'])


trained_model=model.fit(X_train,Y_train,
         epochs=30,
         batch_size=batch,
         steps_per_epoch=X_train.shape[0]//batch,
         validation_steps=X_test.shape[0]//batch,
         validation_data=(X_test,Y_test),
         verbose=2)

test_datagen = ImageDataGenerator()

prediction=model.predict(X_test)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:', score[1])


