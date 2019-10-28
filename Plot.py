from matplotlib import pyplot as plt

def PlotResult(history, metric):
    #print(history.history.keys())
    values = history.history[metric]
    plt.plot(values)
    #plt.plot(history.history['val_loss'])
    plt.title('model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    #plt.ylim(ymin = 0, ymax = 1)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.text(len(values) - (len(values)/10+1), values[-1] + 0.01, '%.4f'%values[-1])
    plt.show()

