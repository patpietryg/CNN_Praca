from keras.preprocessing.image import ImageDataGenerator

def augment_data(X_train, y_train, batch_size_val):
    dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
    dataGen.fit(X_train)
    batches = dataGen.flow(X_train, y_train, batch_size=batch_size_val)
    X_batch, y_batch = next(batches)
    return X_batch, y_batch, dataGen
