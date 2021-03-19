import glob
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# Loading Data
txt_path = r"C:/Users/andyb/GDrive/Education/DataScienceMentorship/Dataset/Additional_Info"
img_path = r"C:/Users/andyb/GDrive/Education/DataScienceMentorship/Dataset/Images" 

img = [file_name for file_name in glob.glob(f'{img_path}/*.jpg')]
txt = [file_name for file_name in glob.glob(f'{txt_path}/*.txt')]

def load_txt(file_name, cols=["file_name","index_label"]):
    txt_file = pd.read_table(file_name, sep='\s+', header=None, names=cols, index_col=False)
    if txt_file.columns[0] == "file_name":
        txt_file["file_name"]= txt_file["file_name"].replace(to_replace='calibrated/', value='', regex=True)
    return txt_file

label_map = load_txt(txt[0], ["index_label","label"]).set_index("index_label").T.to_dict("list")
test_labels = load_txt(txt[2])
train_labels = load_txt(txt[3])
validation_labels = load_txt(txt[4])

# Exploring Data
test_labels.describe()
train_labels.describe()
validation_labels.describe()

fig,ax = plt.subplots(3,3,figsize=(10,10))
plt.suptitle('Test Data')

for i in range(3):
  for j in range(3):
    x=random.randint(1,test_labels.shape[0])
    ax[i,j].imshow(mpimg.imread(f'{img_path}/{test_labels.iloc[x][0]}'))
    ax[i,j].set_title(label_map[test_labels.iloc[x][1]])
    ax[i,j].axis('off')

plt.show()