import json

SpellLabels ={
    "barrier":  0,
    "cleanse":  1,
    "exhaust":  2,
    "flash":    3,
    "ghost":    4,
    "heal":     5,
    "hexflash": 6,
    "ignite":   7,
    "smite":    8,
    "teleport": 9,
}

SpellCooldowns ={
    "barrier":  180,
    "cleanse":  210,
    "exhaust":  210,
    "flash":    300,
    "ghost":    180,
    "heal":     240,
    "hexflash": 100,
    "ignite":   180,
    "smite":    100,
    "teleport": 360, 
}

def LoadJson(folder):
    path = "dataset\\" + folder + "\\labels.json"
    with open(path, 'r') as file:
        data = json.load(file)
    return data
