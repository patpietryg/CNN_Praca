from imblearn.under_sampling import RandomUnderSampler
import numpy as np

def undersample(images, classNo, threshold=250):
    unique_classes, class_counts = np.unique(classNo, return_counts=True)

    # Wybierz klasy, które mają więcej niż threshold zdjęć
    classes_to_undersample = unique_classes[class_counts > threshold]

    # Inicjalizacja RandomUnderSampler
    sampler = RandomUnderSampler(random_state=42)

    # Przechowuje zundersamplingowane dane
    undersampled_images = []
    undersampled_classNo = []

    for class_label in classes_to_undersample:
        # Wybierz indeksy próbek dla danej klasy
        class_indices = np.where(classNo == class_label)[0]

        # Przygotuj dane do zundersamplingowania
        X_class = images[class_indices]
        y_class = classNo[class_indices]
        print(y_class)
        # Zundersamplinguj dane
        X_resampled, y_resampled = sampler.fit_resample(X_class, y_class)

        # Dodaj zundersamplingowane dane do list
        undersampled_images.extend(X_resampled)
        undersampled_classNo.extend(y_resampled)

    # Konwersja na tablice numpy
    undersampled_images = np.array(undersampled_images)
    undersampled_classNo = np.array(undersampled_classNo)

    return undersampled_images, undersampled_classNo
