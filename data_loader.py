import os
import cv2
import numpy as np

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

    return  images, classNo, noOfClasses