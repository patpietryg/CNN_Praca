import numpy as np
from data_loader import load_data
from data_preprocessing import preprocessing, reshape, categorical
from data_augmentation import augment_data
from model_builder import build_model
from model_saver import save_model
from model_plot import draw_plot

path = "myData"
testRatio = 0.1
validationRatio = 0.1
batch_size_val = 20
epochs_val = 15
imageDimesions = (32, 32, 3)

X_train, X_validation, X_test, y_train, y_validation, y_test, noOfClasses = load_data(path, testRatio, validationRatio)

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
