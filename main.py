import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from data_loader import load_data
from data_preprocessing import preprocessing, reshape, categorical
from data_augmentation import augment_data
from model_builder import build_model
from model_saver import save_model
from model_plot import draw_plot

import matplotlib.pyplot as plt

path = "myData"
testRatio = 0.1
validationRatio = 0.1
batch_size_val = 20
epochs_val = 15
imageDimesions = (32, 32, 3)

images, classNo, noOfClasses = load_data(path, testRatio, validationRatio)

class_counts_before_undersampling = np.bincount(classNo)

# # Undersampling danych
# undersample = RandomUnderSampler(sampling_strategy='auto', random_state=42)
# images_resampled, classNo_resampled = undersample.fit_resample(images.reshape(-1, 32*32*3), classNo)
# images_resampled = images_resampled.reshape(-1, 32, 32, 3)

# Sprawdź liczbę przykładów w każdej klasie po undersamplingu
print("Liczba przykładów po undersamplingu:", Counter(classNo))

# Podział danych po undersamplingu
X_train, X_validation, y_train, y_validation = train_test_split(images, classNo, test_size=validationRatio)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=testRatio)

# # Tworzenie wykresu po undersamplingu
# class_counts_after_undersampling = np.bincount(y_train)
# nazwy_klas_after_undersampling = [f'Klasa {i}' for i in range(len(class_counts_after_undersampling))]
#
# plt.figure(figsize=(10, 6))
# plt.bar(nazwy_klas_after_undersampling, class_counts_after_undersampling, color='skyblue')
# plt.xlabel('Nazwa klasy')
# plt.ylabel('Liczba zdjęć')
# plt.title('Ilość zdjęć w poszczególnych klasach po undersamplingu')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

print("Ilość danych")
print("Do trenowania: ")
print("x: ", X_train.shape[0], " y: ", y_train.shape[0])
print("Validation: ")
print("x: ", X_validation.shape[0], " y: ", y_validation.shape[0])
print("Test: ")
print("x: ", X_test.shape[0], " y: ", y_test.shape[0])

X_train, X_validation, X_test = reshape(X_train, X_validation, X_test)

X_batch, y_batch, dataGen = augment_data(X_train, y_train, batch_size_val)

y_train, y_validation, y_test = categorical(y_train, y_test, y_validation, noOfClasses)

model = build_model(imageDimesions, noOfClasses)
print(model.summary())

history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                              None, epochs=epochs_val,
                              validation_data=(X_validation, y_validation), shuffle=1)

draw_plot(history, model, X_test, y_test)

save_model(model, "model_trained.p")
