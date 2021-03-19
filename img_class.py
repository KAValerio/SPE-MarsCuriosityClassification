import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from PIL import Image


txt_path = r"C:/Users/andyb/GDrive/Education/DataScienceMentorship/Dataset/Additional_Info"
img_path = r"C:/Users/andyb/GDrive/Education/DataScienceMentorship/Dataset/Images" #Image.open(img[0])

img = [file_name for file_name in glob.glob(f'{img_path}/*.jpg')]
txt = [file_name for file_name in glob.glob(f'{txt_path}/*.txt')]

def load_txt(file_name, cols=["file_name","index_label"]):
    txt_file = pd.read_table(file_name, sep='\s+', header=None, names=cols, index_col=False)
    if txt_file.columns[0] == "file_name":
        txt_file["file_name"]= txt_file["file_name"].replace(to_replace='calibrated/', value='', regex=True)
    return txt_file

label_map = load_txt(txt[0], ["index_label","label"])
test_labels = load_txt(txt[2])
train_labels = load_txt(txt[3])
validation_labels = load_txt(txt[4])

plt.figure(figsize=(10, 10))
for images, labels in test_labels.iteritems():
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")