import json


def to_int(string):
    try:
        return int(string)
    except ValueError:
        return 0


def load_json(folder):
    path = "data\\image\\dataset\\" + folder + "\\labels.json"
    with open(path, 'r') as file:
        labels = json.load(file)
    return labels
