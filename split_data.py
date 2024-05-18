import os
from sklearn.model_selection import train_test_split
import shutil
import os

PATH = '.../fisheye8k/train' # path to train
PATH_create = '.../fisheye8k/' # path to create folder

os.mkdir(PATH_create + "train - Copy")
os.mkdir(PATH_create + "val")
os.mkdir(PATH_create + "train - Copy/images")
os.mkdir(PATH_create + "train - Copy/labels")
os.mkdir(PATH_create + "val/images")
os.mkdir(PATH_create + "val/labels")

# Check if the directory exists
if os.path.exists(PATH + '/images'):
    images = [os.path.join(PATH + '/images', x) for x in os.listdir(PATH + '/images')]
else:
    print("Directory does not exist:", PATH + '/images')
    images = []

# Read images and annotations
images = [os.path.join(PATH + '/images', x) for x in os.listdir(PATH + '/images')]
annotations = [os.path.join(PATH + '/labels', x) for x in os.listdir(PATH + '/labels') if x[-3:] == "txt"]

images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits 
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
# Ensure consistent number of samples
train_images = train_images[:len(train_annotations)]
val_images = val_images[:len(val_annotations)]

#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
move_files_to_folder(train_images, PATH_create + 'train - Copy/images')
move_files_to_folder(val_images, PATH_create + 'val/images')
move_files_to_folder(train_annotations, PATH_create + 'train - Copy/labels')
move_files_to_folder(val_annotations, PATH_create + 'val/labels')

shutil.rmtree('.../fisheye8k/train')
os.rename(PATH_create + 'train - Copy', PATH_create + 'train')