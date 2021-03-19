import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def LoadDat(path, grayscale=False):
    """
    :param str path: path to the image from 'calibrated'. Function adds 'msl-images/'
    :param bool grayscale: Set to true if images are to be gray-scaled
    :returns:
        - ot_X - numpy array of the pictures of size (NxHxWxC) where C is the color. Returns (NxHxW) if grayscale.
        - ot_y - labels associated with the images of size (N)
    """
    if grayscale:
        imcol = 'L'
        dtypearr = float
    else:
        imcol = 'RGB'
        dtypearr = int

    ot = np.genfromtxt(path, dtype=str)
    ot_paths = ot[:, 0]
    ot_Y = ot[:, 1]  # Training Labels

    ot_X = []
    for name in ot_paths:
        im = Image.open(f'msl-images/{name}').resize((256, 256), Image.ANTIALIAS).convert(mode=imcol)
        ar = np.array(im, dtype=dtypearr)
        ot_X.append(ar)

    ot_X = np.asarray(ot_X)
    return ot_X, ot_Y


def plot_img(dataset, labels, title="Sample Data"):
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    plt.suptitle(title)
    np.random.seed(42)  # comment this line out for random seed
    for i in range(3):
        for j in range(3):
            x = np.random.randint(0, dataset.shape[0])
            ax[i, j].imshow(dataset[x])
            ax[i, j].set_title(label_meaning[int(labels[x]), 1])
            ax[i, j].axis('off')
    plt.show()


# Load the label defenitions. Note that column 1 matches the index.
lab_path = 'msl-images/msl_synset_words-indexed.txt'
label_meaning = np.genfromtxt(lab_path, dtype=str, delimiter="  ", usecols=(0, -1))
label_meaning = np.char.strip(label_meaning)

# Load the training data (Used for training models)
train_path = 'msl-images/train-calibrated-shuffled.txt'
train_X, train_Y = LoadDat(train_path)

# Load the validation data (Used to compare models)
val_path = 'msl-images/test-calibrated-shuffled.txt'
val_X, val_Y = LoadDat(val_path)

# Load the testing data (Used for testing the FINAL model)
test_path = 'msl-images/test-calibrated-shuffled.txt'
test_X, test_Y = LoadDat(test_path)

# Make Figure
plot_img(test_X, test_Y, "Test Data")

# Class Imbalances
class_imbalance = Counter(np.concatenate((train_Y, test_Y, val_Y)))
class_imbalance = dict(sorted(class_imbalance.items(), key=lambda item: item[1], reverse=True))