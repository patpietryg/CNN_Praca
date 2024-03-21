import numpy as np
import pickle
import os
import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Parametry
path = "myData"  # folder ze znakami drogowymi
labelFile = 'labels.csv'  # folder z etykietami
batch_size_val = 20
steps_per_epoch_val = 1000
epochs_val = 15
imageDimesions = (32, 32, 3)
testRatio = 0.1  # % na testy
validationRatio = 0.1  # % na walidacje

# Pobieranie danych, zdjęć z folderu
images = []
classNo = []
counter = 0
myList = os.listdir(path)
print("Znaleziona ilość rodzaji znaków drogowych:", len(myList))
noOfClasses = len(myList)
print("Pobieranie...")
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(counter))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(counter) + "/" + y)
        images.append(curImg)
        classNo.append(counter)
    print(counter, end=" ")
    counter += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

# Podzielenie danych
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

# Ilość danych
print("Ilość danych")
print("Do trenowania: ")
print("x: ", X_train.shape[0], " y: ", y_train.shape[0])
print("Validation: ")
print("x: ", X_validation.shape[0], " y: ", y_validation.shape[0])
print("Test: ")
print("x: ", X_test.shape[0], " y: ", y_test.shape[0])

# Wczytywanie etykiet
data = pd.read_csv(labelFile, encoding='latin1')

# Przetwarzanie obrazów
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)  # zmiana na skale szarości
    img = equalize(img)  # ustandaryzowanie oświetlenia
    img = img / 255  # wartosc od 0 do 1
    return img

# zamiana obrazów na wartości od 0 do 1
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# Zmiana kształu listy na głębokość =1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Powiększenie obrazów dla większego uogólnienia
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, 20)
X_batch, y_batch = next(batches)


y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)


# Budowanie sieci CNN
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500
    model = Sequential()

    model.add(Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1)))
    model.add(BatchNormalization())  # Dodajemy warstwę Batch Normalization
    model.add(Activation("relu"))  # Dodajemy aktywację ReLU

    model.add(Conv2D(no_Of_Filters, size_of_Filter))
    model.add(BatchNormalization())  # Dodajemy warstwę Batch Normalization
    model.add(Activation("relu"))  # Dodajemy aktywację ReLU
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add(Conv2D(no_Of_Filters // 2, size_of_Filter2))
    model.add(BatchNormalization())  # Dodajemy warstwę Batch Normalization
    model.add(Activation("relu"))  # Dodajemy aktywację ReLU

    model.add(Conv2D(no_Of_Filters // 2, size_of_Filter2))
    model.add(BatchNormalization())  # Dodajemy warstwę Batch Normalization
    model.add(Activation("relu"))  # Dodajemy aktywację ReLU
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes))
    model.add(BatchNormalization())  # Dodajemy warstwę Batch Normalization
    model.add(Activation("relu"))  # Dodajemy aktywację ReLU
    model.add(Dropout(0.5))

    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Trening
model = myModel()
print(model.summary())
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                              None, epochs=epochs_val,
                              validation_data=(X_validation, y_validation), shuffle=1)

# Zapis jako pickle
pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0)