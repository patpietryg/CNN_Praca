import cv2
from tensorflow.keras.utils import to_categorical

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

def reshape(X_train, X_validation, X_test):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    return X_train, X_validation, X_test

def categorical(y_train, y_test, y_validation, noOfClasses):
    y_train = to_categorical(y_train, noOfClasses)
    y_validation = to_categorical(y_validation, noOfClasses)
    y_test = to_categorical(y_test, noOfClasses)
    return y_train, y_validation, y_test