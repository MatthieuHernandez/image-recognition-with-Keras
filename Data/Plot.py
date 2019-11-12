from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from data.regression import Regression


class Index(object):

    def __init__(self, dataset, jump):
        self.fig, self.axe = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        print("toto")
        self.index = 0
        self.dataset = dataset
        self.jump = jump
        self.__execute()

    def next(self, _event):
        self.index += self.jump
        self.__execute()

    def prev(self, _event):
        self.index -= self.jump
        self.__execute()

    def __execute(self):
        self.axe.set_title("{0}    index={1}".format(
            self.dataset[1][self.index], self.index))
        self.axe.imshow(
            self.dataset[0][self.index].reshape(20, 20), cmap='gray')
        plt.draw()


def display_set(dataset, jump=1):

    callback = Index(dataset, jump)
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    plt.show()


def main():
    print("Start")
    #trainSet = Regression.load_set(["train", "train_fake", "test_hard", "train_auto-generated"])
    set_to_display = Regression.load_set(["train_auto-generated"])
    display_set(set_to_display)
    print("End")


if __name__ == "__main__":
    main()
