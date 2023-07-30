Breast Cancer Classification – Objective
To build a breast cancer classifier on an IDC dataset that can accurately classify a histology image as benign or malignant.
Breast Cancer Classification – About the Python Project
In this project in python, we’ll build a classifier to train on 80% of a breast cancer histology image dataset. Of this, we’ll keep 10% of the data for validation. Using Keras, we’ll define a CNN (Convolutional Neural Network), call it CancerNet, and train it on our images. We’ll then derive a confusion matrix to analyze the performance of the model.
IDC is Invasive Ductal Carcinoma; cancer that develops in a milk duct and invades the fibrous or fatty breast tissue outside the duct; it is the most common form of breast cancer forming 80% of all breast cancer diagnoses. And histology is the study of the microscopic structure of tissues.
The Dataset

DATASET:- https://drive.google.com/open?id=1nEkiRNIdYUSi0Eyci19KceLJjObGB25m

We’ll use the IDC_regular dataset (the breast cancer histology image dataset) from Kaggle. This dataset holds 2,77,524 patches of size 50×50 extracted from 162 whole mount slide images of breast cancer specimens scanned at 40x. Of these, 1,98,738 test negative and 78,786 test positive with IDC. The dataset is available in public domain and you can download it here. You’ll need a minimum of 3.02GB of disk space for this.
Filenames in this dataset look like this:
8863_idx5_x451_y1451_class0
Here, 8863_idx5 is the patient ID, 451 and 1451 are the x- and y- coordinates of the crop, and 0 is the class label (0 denotes absence of IDC).
Prerequisites
You’ll need to install some python packages to be able to run this advanced python project. You can do this with pip-
pip install numpy opencv-python pillow tensorflow keras imutils scikit-learn matplotlib
Steps for Advanced Project in Python – Breast Cancer Classification
1. Download this zip. Unzip it at your preferred location, get there.
Screenshot:
![image](https://github.com/varshiniargula/Breast-Cancer-Classification/assets/133252654/2d03f3e0-b8ff-49a2-8acd-bc4df28dd684)

2. Now, inside the inner breast-cancer-classification directory, create directory datasets- inside this, create directory original:
mkdir datasets
mkdir datasets\original
3. Download the dataset.
4. Unzip the dataset in the original directory. To observe the structure of this directory, we’ll use the tree command:
cd breast-cancer-classification\breast-cancer-classification\datasets\original
tree
Output Screenshot:
![image](https://github.com/varshiniargula/Breast-Cancer-Classification/assets/133252654/6176647c-0aca-48b4-8854-c594758cad0b)

We have a directory for each patient ID. And in each such directory, we have the 0 and 1 directories for images with benign and malignant content.
config.py:
This holds some configuration we’ll need for building the dataset and training the model. You’ll find this in the cancernet directory.
import os
INPUT_DATASET = "datasets/original"
BASE_PATH = "datasets/idc"
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
Screenshot:
![image](https://github.com/varshiniargula/Breast-Cancer-Classification/assets/133252654/18bf5c98-cf84-4328-95eb-09409dd11d25)

Here, we declare the path to the input dataset (datasets/original), that for the new directory (datasets/idc), and the paths for the training, validation, and testing directories using the base path. We also declare that 80% of the entire dataset will be used for training, and of that, 10% will be used for validation.

build_dataset.py:
This will split our dataset into training, validation, and testing sets in the ratio mentioned above- 80% for training (of that, 10% for validation) and 20% for testing. With the ImageDataGenerator from Keras, we will extract batches of images to avoid making space for the entire dataset in memory at once.

Screenshot:
![image](https://github.com/varshiniargula/Breast-Cancer-Classification/assets/133252654/8747c9b3-3aa8-408d-b558-fcb54e0b66b2)

In this, we’ll import from config, imutils, random, shutil, and os. We’ll build a list of original paths to the images, then shuffle the list. Then, we calculate an index by multiplying the length of this list by 0.8 so we can slice this list to get sublists for the training and testing datasets. Next, we further calculate an index saving 10% of the list for the training dataset for validation and keeping the rest for training itself.
Now, datasets is a list with tuples for information about the training, validation, and testing sets. These hold the paths and the base path for each. For each setType, path, and base path in this list, we’ll print, say, ‘Building testing set’. If the base path does not exist, we’ll create the directory. And for each path in originalPaths, we’ll extract the filename and the class label. We’ll build the path to the label directory(0 or 1)- if it doesn’t exist yet, we’ll explicitly create this directory. Now, we’ll build the path to the resulting image and copy the image here- where it belongs.
5. Run the script build_dataset.py:
py build_dataset.py
Output Screenshot:
![image](https://github.com/varshiniargula/Breast-Cancer-Classification/assets/133252654/c3612712-74f3-4b23-9c79-9fa2c556353a)


Screenshot:
![image](https://github.com/varshiniargula/Breast-Cancer-Classification/assets/133252654/10981d32-517c-4a94-a656-ee58ca36b55a)

Screenshot:
![image](https://github.com/varshiniargula/Breast-Cancer-Classification/assets/133252654/cc42c388-8157-49f5-ad89-bf6330c70664)

We use the Sequential API to build CancerNet and SeparableConv2D to implement depthwise convolutions. The class CancerNet has a static method build that takes four parameters- width and height of the image, its depth (the number of color channels in each image), and the number of classes the network will predict between, which, for us, is 2 (0 and 1).
In this method, we initialize model and shape. When using channels_first, we update the shape and the channel dimension.
Now, we’ll define three DEPTHWISE_CONV => RELU => POOL layers; each with a higher stacking and a greater number of filters. The softmax classifier outputs prediction percentages for each class. In the end, we return the model.

Screenshot:
![image](https://github.com/varshiniargula/Breast-Cancer-Classification/assets/133252654/e69734b4-a84e-4af8-9ec1-b707999b08d2)

Screenshot:
![image](https://github.com/varshiniargula/Breast-Cancer-Classification/assets/133252654/86cccd3d-e9bf-4f7e-80cd-47ab33b43e85)

Screenshot:
![image](https://github.com/varshiniargula/Breast-Cancer-Classification/assets/133252654/333fbbae-6c31-4e60-b5c7-5e7df13e5f2f)

In this script, first, we set initial values for the number of epochs, the learning rate, and the batch size. We’ll get the number of paths in the three directories for training, validation, and testing. Then, we’ll get the class weight for the training data so we can deal with the imbalance.
Now, we initialize the training data augmentation object. This is a process of regularization that helps generalize the model. This is where we slightly modify the training examples to avoid the need for more training data. We’ll initialize the validation and testing data augmentation objects.
We’ll initialize the training, validation, and testing generators so they can generate batches of images of size batch_size. Then, we’ll initialize the model using the Adagrad optimizer and compile it with a binary_crossentropy loss function. Now, to fit the model, we make a call to fit_generator().
We have successfully trained our model. Now, let’s evaluate the model on our testing data. We’ll reset the generator and make predictions on the data. Then, for images from the testing set, we get the indices of the labels with the corresponding largest predicted probability. And we’ll display a classification report.
Now, we’ll compute the confusion matrix and get the raw accuracy, specificity, and sensitivity, and display all values. Finally, we’ll plot the training loss and accuracy.
Output Screenshot:
![image](https://github.com/varshiniargula/Breast-Cancer-Classification/assets/133252654/13db5fc7-d34b-4315-af5f-f71a0b392cc6)

Output Screenshot:
![image](https://github.com/varshiniargula/Breast-Cancer-Classification/assets/133252654/9b72c65a-d835-476b-8a89-020813b7b433)

Output:
![image](https://github.com/varshiniargula/Breast-Cancer-Classification/assets/133252654/3bd16746-6ff7-4c73-ae4e-905f1958139f)

Summary
In this project in python, we learned to build a breast cancer classifier on the IDC dataset (with histology images for Invasive Ductal Carcinoma) and created the network CancerNet for the same. We used Keras to implement the same. Hope you enjoyed this Python project.

