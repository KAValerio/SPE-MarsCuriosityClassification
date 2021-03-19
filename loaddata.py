import numpy as np
from PIL import Image

# Load the label defenitions
lab_path = 'msl-images/msl_synset_words-indexed.txt'
labs = np.genfromtxt(lab_path, dtype=str, delimiter="  ", usecols=(0, -1))
labs = np.char.strip(labs)

# Load the training data
train_path = 'msl-images/train-calibrated-shuffled.txt'
train = np.genfromtxt(train_path, dtype=str)
train_paths = train[:, 0]
train_Y = train[:, 1]             #Training Labels


