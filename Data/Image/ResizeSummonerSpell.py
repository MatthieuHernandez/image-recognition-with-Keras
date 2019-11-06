import glob
from PIL import Image
from ImageGenerator import CleanFolder

def ResizeAllImages():
    newSize = 20
    path = "dataset\\summoner_spells\\*.png"
    for fileName in glob.glob(path):
        img = Image.open(fileName)
        img = img.resize((newSize,newSize), Image.ANTIALIAS)
        fileName = "dataset\\summoner_spells\\resized\\" + fileName.split('\\')[-1]
        img.save(fileName)

if __name__ == "__main__":
    print("Start")

    CleanFolder("dataset\\summoner_spells\\resized")
    ResizeAllImages()

    print("End")