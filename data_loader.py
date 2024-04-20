import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path, testRatio, validationRatio):
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
    # X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
    # X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
    #
    # return X_train, X_validation, X_test, y_train, y_validation, y_test, noOfClasses

    return  images, classNo, noOfClasses