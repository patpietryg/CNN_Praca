import matplotlib.pyplot as plt

def draw_plot(history, model, X_test, y_test):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['trening', 'walidacja'])
    plt.title('strata')
    plt.xlabel('epoka')
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['trening', 'walidacja'])
    plt.title('Dokładność')
    plt.xlabel('epoka')
    plt.show()
    wynik = model.evaluate(X_test, y_test, verbose=0)
    print('Wynik testu:', wynik[0])
    print('Dokładność testu:', wynik[1])
