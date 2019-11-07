import glob
from PIL import Image
from data.image.generator import clean_folder


def resize_all_images():
    newSize = 20
    path = "dataset\\summoner_spells\\*.png"
    for fileName in glob.glob(path):
        img = Image.open(fileName)
        img = img.resize((newSize, newSize), Image.ANTIALIAS)
        fileName = "dataset\\summoner_spells\\resized\\" + \
            fileName.split('\\')[-1]
        img.save(fileName)


def main():
    print("Start")
    clean_folder("dataset\\summoner_spells\\resized")
    resize_all_images()
    print("End")


if __name__ == "__main__":
    main()
