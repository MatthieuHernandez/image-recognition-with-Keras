import numpy as np

def GenerateImages(folder):
    
    path =  "dataset\\" + folder + "\\inputs\\"
    
    for n in range(0, 2):
        for angle in range(0, 360):
            img = CreateImage()
        #AddNoise()
        #Addline()
        #Rotate(angle)
        #Addline()


def CreateImage ():
    img = np.zeros((20, 20, 3))
    return img





















if __name__ == "__main__":
    print("Start")


    GenerateImages("train_fake_auto-generated")
    
    print("End")
