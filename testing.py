import numpy as np
import cv2
import pickle

# Ustawienie kamery
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# import modelu
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img


def getClassName(classNo):
    if classNo == 0:
        return 'Ograniczenie prędkości (20 km/h)'
    elif classNo == 1:
        return 'Ograniczenie prędkości (30 km/h)'
    elif classNo == 2:
        return 'Ograniczenie prędkości (50 km/h)'
    elif classNo == 3:
        return 'Ograniczenie prędkości (60 km/h)'
    elif classNo == 4:
        return 'Ograniczenie prędkości (70 km/h)'
    elif classNo == 5:
        return 'Ograniczenie prędkości (80 km/h)'
    elif classNo == 6:
        return 'Koniec ograniczenia prędkości (80 km/h)'
    elif classNo == 7:
        return 'Ograniczenie prędkości (100 km/h)'
    elif classNo == 8:
        return 'Ograniczenie prędkości (120 km/h)'
    elif classNo == 9:
        return 'Zakaz wyprzedzania'
    elif classNo == 10:
        return 'Zakaz wyprzedzania dla pojazdów powyżej 3.5 tony'
    elif classNo == 11:
        return 'Pierwszeństwo przejazdu na następnym skrzyżowaniu'
    elif classNo == 12:
        return 'Droga o pierwszeństwie przejazdu'
    elif classNo == 13:
        return 'Ustąp pierwszeństwa'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'Brak pojazdów'
    elif classNo == 16:
        return 'Zakaz wjazdu pojazdów o masie powyżej 3.5 tony'
    elif classNo == 17:
        return 'Zakaz wjazdu'
    elif classNo == 18:
        return 'Ogólna ostrożność'
    elif classNo == 19:
        return 'Niebezpieczny zakręt w lewo'
    elif classNo == 20:
        return 'Niebezpieczny zakręt w prawo'
    elif classNo == 21:
        return 'Podwójny zakręt'
    elif classNo == 22:
        return 'Nierówna droga'
    elif classNo == 23:
        return 'Śliska droga'
    elif classNo == 24:
        return 'Droga zwęża się po prawej stronie'
    elif classNo == 25:
        return 'Roboty drogowe'
    elif classNo == 26:
        return 'Sygnalizacja świetlna'
    elif classNo == 27:
        return 'Piesi'
    elif classNo == 28:
        return 'Przejście dla dzieci'
    elif classNo == 29:
        return 'Przejście dla rowerzystów'
    elif classNo == 30:
        return 'Uwaga na lód/śnieg'
    elif classNo == 31:
        return 'Przejście dzikich zwierząt'
    elif classNo == 32:
        return 'Koniec wszystkich ograniczeń prędkości i zakazów wyprzedzania'
    elif classNo == 33:
        return 'Skręć w prawo'
    elif classNo == 34:
        return 'Skręć w lewo'
    elif classNo == 35:
        return 'Tylko prosto'
    elif classNo == 36:
        return 'Prosto albo w prawo'
    elif classNo == 37:
        return 'Prosto albo w lewo'
    elif classNo == 38:
        return 'Jedź prawo'
    elif classNo == 39:
        return 'Jedź lewo'
    elif classNo == 40:
        return 'Rondo obowiązkowe'
    elif classNo == 41:
        return 'Koniec zakazu wyprzedzania'
    elif classNo == 42:
        return 'Koniec zakazu wyprzedzania dla pojazdów o masie powyżej 3.5 tony'


# Lista przechowująca pięć klatek
frames_buffer = []

# Dane ostatniej klasyfikacji
last_class_index = None
last_probability_value = None

while True:
    # wczytanie obrazu
    success, imgOrignal = cap.read()

    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # Dodawanie kolejnych klatek do bufora
    frames_buffer.append(img)

    # Sprawdzenie, czy mamy już pięć klatek w buforze
    if len(frames_buffer) == 5:
        # Obliczenie średniej klatki
        averaged_frame = np.mean(frames_buffer, axis=0)

        # Wykonanie klasyfikacji
        predictions = model.predict(averaged_frame)
        classIndex = np.argmax(predictions)
        probabilityValue = predictions[0, classIndex]

        # Przypisanie danych ostatniej klasyfikacji
        last_class_index = classIndex
        last_probability_value = probabilityValue

        # Wyczyszczenie bufora
        frames_buffer = []

    # Wyświetlanie danych ostatniej klasyfikacji
    if last_class_index is not None and last_probability_value is not None:
        if last_probability_value > threshold:
            class_name = getClassName(last_class_index)
            cv2.putText(imgOrignal, str(last_class_index) + " " + class_name, (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOrignal, str(round(last_probability_value * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2,
                        cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()