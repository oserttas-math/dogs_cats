import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm


# Paths
PROJ_ROOT = os.path.join(os.pardir)
path_training = os.path.join(PROJ_ROOT, "data", "raw", "train/")
path_testing  = os.path.join(PROJ_ROOT, "data", "raw", "test/")


# Labeling the dataset
def label_img(img):
    word_label = img.split(".")[-3]
    if word_label == 'cat':
        return [1,0]
    elif word_label == 'dog':
        return [0,1]
    
    
    
# Create the training data
# Create the training data
def create_training_data():
    training_data = []
    training_labels = []
    # tqdm is only used for interactive loading
    # loading the training data
    for img in tqdm(os.listdir(path_training)):
        
        # label of the image
        label = label_img(img)
        
        # path to the image
        path = os.path.join(path_training, img)
        
        # load the image from the path and convert it to grayscale for simplicity
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # resize the image
        img = cv2.resize(img, (50, 50))
        
        # final step-forming the training data list wiht numpy array of images
        training_data.append(img)
        training_labels.append(label)
        
    # shuffling of the training data to preserve the random state of our data
    shuffle(training_data)
    
    # randomly choose 1/5 of training set and call it validation set
    validation_set = training_data[:len(training_data)//5]
    validation_labels = training_labels[:len(training_labels)//5]
    training_labels = training_labels[len(training_labels)//5:]
    training_set   = training_data[len(training_data)//5:]
    
    # save the trained data for further uses if needed
#     np.save(os.path.join(PROJ_ROOT,'data', 'interim','training_data'), training_set)
#     np.save(os.path.join(PROJ_ROOT,'data', 'interim','validation_data'), validation_set)
#     np.save(os.path.join(PROJ_ROOT,'data', 'interim','validation_labels'), validation_labels)
#     np.save(os.path.join(PROJ_ROOT,'data', 'interim','training_labels'), training_labels)
    return validation_set, validation_labels, training_set, training_labels
        




# Convert the test data as well
def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(path_testing)):
        # path to the image
        path = os.path.join(path_testing, img)
        
        img_num = img.split(".")[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))
        testing_data.append([np.array(img),img_num])
    shuffle(testing_data)
    np.save(os.path.join(PROJ_ROOT, "data", "interim", "test_data"), testing_data)
    return testing_data



