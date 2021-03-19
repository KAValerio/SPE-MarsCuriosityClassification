import numpy as np
from PIL import Image

# Load the label defenitions
lab_path = 'msl-images/msl_synset_words-indexed.txt'
labs = np.genfromtxt(lab_path, dtype=str, delimiter="  ", usecols=(0, -1))
labs = np.char.strip(labs)


def LoadDat(path):
    ot = np.genfromtxt(path, dtype=str)
    ot_paths = ot[:, 0]
    ot_Y = ot[:, 1]  # Training Labels
    ot_X = np.array([np.array(Image.open('msl-images/' + name)) for name in ot_paths], dtype=object)
    return ot_X, ot_Y

# Load the training data (Used for training models)
# ____X: numpy list where each row is an image, i.e. train_X[0].shape = (191, 255, 3), train_X.shape = (3746,)
train_path = 'msl-images/train-calibrated-shuffled.txt'
train_X, train_Y = LoadDat(train_path)

# Load the validation data (Used to compare models)
val_path = 'msl-images/test-calibrated-shuffled.txt'
val_X, val_Y = LoadDat(val_path)

# Load the testing data (Used for testing the FINAL model)
test_path = 'msl-images/test-calibrated-shuffled.txt'
test_X, test_Y = LoadDat(test_path)

