import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def LoadData(path, grayscale=False):
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
        im = Image.open(
            f'msl-images/{name}').resize((256, 256), Image.ANTIALIAS).convert(mode=imcol)
        ar = np.array(im, dtype=dtypearr)
        ot_X.append(ar)

    ot_X = np.asarray(ot_X)
    return ot_X, ot_Y


def plotImg(dataset, labels, title="Sample Data"):
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    plt.suptitle(title)
    # np.random.seed(42)  # comment this line out for random seed
    for i in range(3):
        for j in range(3):
            x = np.random.randint(0, dataset.shape[0])
            ax[i, j].imshow(dataset[x])
            ax[i, j].set_title(label_meaning[int(labels[x]), 1])
            ax[i, j].axis('off')
    plt.show()


def plotBar(dataset, label, loc='center', relative=True):
    counter_dict = dict(Counter(dataset))
    sorted_dict = dict(sorted(counter_dict.items(), key=lambda x: int(x[0])))
    dataset_dict = dict(zip(list(dict(label).values()),
                        list(sorted_dict.values())))

    width = 0.25
    if loc == 'left':
        n = -0.2
    elif loc == 'right':
        n = 0.2
    elif loc == 'center':
        n = 0

    if relative == True:
        # plot as percentage
        counts_dict = dataset_dict.copy()
        counts_dict.update(
            {n: round((100 * dataset_dict[n])/sum(dataset_dict.values()), 2) for n in dataset_dict.keys()})
        counts = list(counts_dict.values())
        counts_df = pd.DataFrame.from_dict(
            counts_dict, orient='index').reset_index()
        counts_df.columns = ['labels', 'count (%)']
        ylabel_text = '% count'
    else:
        # plot as counts
        counts = list(dataset_dict.values())
        counts_df = pd.DataFrame.from_dict(
            dataset_dict, orient='index').reset_index()
        counts_df.columns = ['labels', 'count']
        ylabel_text = 'count'

    xtemp = np.arange(len(dataset_dict.keys()))

    plt.bar(xtemp + n, counts, align='center', alpha=.7, width=width)
    plt.xticks(xtemp + n, counts_df['labels'], rotation=90)
    plt.title('Class Imbalance', fontsize=14)
    plt.xlabel('labels')
    plt.ylabel(ylabel_text)
    plt.tick_params(axis='x', labelsize=10)

    return counts_df


def imgToVector(image, grayscale=False):
    if grayscale == True:
        length, height = image.shape
        return image.reshape((length * height, ))
    else:
        length, height, depth = image.shape
        return image.reshape((length * height * depth, ))


# Load the label defenitions. Note that column 1 matches the index.
lab_path = 'msl-images/msl_synset_words-indexed.txt'
label_meaning = np.genfromtxt(
    lab_path, dtype=str, delimiter="  ", usecols=(0, -1))
label_meaning = np.char.strip(label_meaning)

gs = False   # Grayscale
# Load the training data (Used for training models)
train_path = 'msl-images/train-calibrated-shuffled.txt'
train_X, train_Y = LoadData(train_path, grayscale=gs)

# Load the validation data (Used to compare models)
val_path = 'msl-images/val-calibrated-shuffled.txt'
val_X, val_Y = LoadData(val_path, grayscale=gs)

# Load the testing data (Used for testing the FINAL model)
test_path = 'msl-images/test-calibrated-shuffled.txt'
test_X, test_Y = LoadData(test_path, grayscale=gs)

# Show images (9 samples)
plotImg(train_X, train_Y, "Train Data")
plotImg(test_X, test_Y, "Test Data")
plotImg(val_X, val_Y, "Validation Data")

# Class Imbalances
class_imbalance = Counter(np.concatenate((train_Y, test_Y, val_Y)))
class_imbalance = dict(sorted(class_imbalance.items(),
                       key=lambda item: item[1], reverse=True))

plt.figure(figsize=(10, 6))
val_count = plotBar(val_Y, label_meaning, loc="left", relative=False)
test_count = plotBar(test_Y, label_meaning, loc="right", relative=False)
train_count = plotBar(train_Y, label_meaning, loc="center", relative=False)
plt.legend([
    'Validation ({0} photos)'.format(sum(dict(Counter(val_Y)).values())),
    'Test ({0} photos)'.format(sum(dict(Counter(test_Y)).values())),
    'Train ({0} photos)'.format(sum(dict(Counter(train_Y)).values()))
])
plt.tight_layout()

total_count = pd.merge(train_count, test_count,
                       how='outer',
                       on='labels',
                       suffixes=('_train', '_test')) \
    .merge(val_count,
           how='outer',
           on='labels',
           suffixes=('_test', '_val'))

# Vectorize images
# train_X_vector = [imgToVector(image, grayscale=True) for image in train_X]
# test_X_vector = [imgToVector(image, grayscale=True) for image in test_X]
# val_X_vector = [imgToVector(image, grayscale=True) for image in val_X]

# Models
#randomForest = RandomForestClassifier()
#randomForest.fit(train_X_vector, train_Y)

# Predictions
#prediction_Y = randomForest.predict(val_X_vector)
#print("Accuracy:", round(accuracy_score(val_Y, prediction_Y), 2))


from skimage.feature import hog
from sklearn import metrics
from sklearn.linear_model import SGDClassifier


def doHOG(data, ppc=12, cpb=3, verbose=True):
    hogs = []
    hogimage = []
    for i in range(data.shape[0]):
        im = train_X[0]
        h, him = hog(im, visualize=True,
                     pixels_per_cell=(ppc, ppc),
                     cells_per_block=(cpb, cpb),
                     block_norm="L2-Hys")
        hogs.append(h)
        hogimage.append(him)
        if verbose:
            if i % round(data.shape[0] / 10) == 0:
                print(f"    {round(i * 100 / data.shape[0])}%")
    return hogs, hogimage

ppc = 9  # HOG parameter (pixels per cell)
cpb = 3  # HOG parameter (cells per block)
print("Starting HOG (Training)")
hogs_train, hogimage_train = doHOG(train_X, ppc=ppc, cpb=cpb)
print("Starting HOG (Testing)")
hogs_test, hogimage_test = doHOG(test_X, ppc=ppc, cpb=cpb)
print("Starting HOG (Validation)")
hogs_valid, hogimage_valid = doHOG(val_X, ppc=ppc, cpb=cpb)


print("Model Creation")
sgd_clf = SGDClassifier(max_iter=5000, tol=1e-5, penalty='elasticnet', n_jobs=-2, loss='squared_hinge', verbose=False)
sgd_clf.fit(np.array(hogs_train), train_Y)
print("Prediction")
y_pred = sgd_clf.predict(np.array(hogs_test))
print('Model Metrics:')
print(metrics.classification_report(test_Y, y_pred))


plt.figure()
plt.imshow(hogimage_train[0])
plt.show()

plt.figure()
plt.imshow(train_X[0])
plt.show()