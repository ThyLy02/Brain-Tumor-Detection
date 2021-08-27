import matplotlib.pyplot as plt
plt.style.use('dark_background')


def plot_model(history):
    x = plt.plot(history.history['loss'])
    y = plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Test', 'Validation'], loc='upper right')
    return plt.show()