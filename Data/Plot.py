from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from Regression import *

fig, ax = plt.subplots()

class Index(object):
    
    def __init__(self, dataSet, jump):
        self.index = 0
        self.dataSet = dataSet
        self.jump = jump
        self.__execute()

    def next(self, _event):
        self.index += self.jump
        self.__execute()

    def prev(self, _event):
        self.index -= self.jump
        self.__execute()

    def __execute(self):
        ax.set_title("{0}    index={1}".format(self.dataSet[1][self.index], self.index))
        ax.imshow(self.dataSet[0][self.index].reshape(20, 20), cmap='gray')
        plt.draw()

        
def displaySet(dataSet, jump = 1):

    plt.subplots_adjust(bottom=0.2)
    callback = Index(dataSet, jump)
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    plt.show()


if __name__ == "__main__":
    print("Start")
    
    #trainSet = Regression.LoadSet(["train", "train_fake", "test_hard", "train_auto-generated"])
    setToDisplay = Regression.LoadSet(["train_auto-generated"])

    #trainSet = Regression.LoadSet(["train"])
    displaySet(setToDisplay)

    print("End")
