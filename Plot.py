from matplotlib import pyplot as plt

def PlotResultLoss(history):
    #print(history.history.keys())
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(ymin = 0, ymax = 1)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def PlotResultAccuracy(history):
    #print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim(ymin = 0, ymax = 1)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
