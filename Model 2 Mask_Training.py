#import
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

#set initial learning rate, epochs, and batch size
InitLR = 0.0001
Epochs = 15
BatS = 32

DIRECTORY = r"C:\Users\sundy\PycharmProjects\FaceMaskDetection"
CATEGORIES = ["dataset/with_mask2", "dataset/without_mask2"]

print("Loading images")

#data. for image array
data = []
#labels. label of the images with and without
labels = []

#loop categories with and without mask
#listdir list all images in the directory
#join join path of the image with the corresponding image
#load_img load all image size to 224 224 for model
#img_to_array convert img to array
#preprocess_input for mobilenet model
#data_append  all the img array into the data list
#labels_append all the labels into the labels list. with and without mask
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)
    	data.append(image)
    	labels.append(category)

#convert with and without mask to categorical variables
#and convert them into numpy(np) array
LabBin = LabelBinarizer()
labels = LabBin.fit_transform(labels)
labels = to_categorical(labels)
data = np.array(data, dtype="float32")
labels = np.array(labels)


#spliting training and testing data
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)



#Create many images from a images by changing some of its properties
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


#MobileNetV2 load model. base model
bModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))


#head model that will be on top of the base model
#pooling
#flattening
#layers
hModel = bModel.output
hModel = AveragePooling2D(pool_size=(7, 7))(hModel)
hModel = Flatten(name="flatten")(hModel)
hModel = Dense(128, activation="relu")(hModel)
hModel = Dropout(0.5)(hModel)
hModel = Dense(2, activation="softmax")(hModel)

#This will be set on top of base and be the model that is going to be train
model = Model(inputs=bModel.input, outputs=hModel)

#loop all layers in the base model and freezing them from the first training
for layer in bModel.layers:
	layer.trainable = False

#compiling the model
#optimzer
opt = Adam(lr=InitLR, decay=InitLR / Epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")

#train the head
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BatS),
	steps_per_epoch=len(trainX) // BatS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BatS,
	epochs=Epochs)

#make predictions on the testing set
print("[INFO] evaluating network...")
predM = model.predict(testX, batch_size=BatS)

#for each image in the testing set we need to find the index of the
#label with corresponding largest predicted probability
predM = np.argmax(predM, axis=1)

#show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predM,
	target_names=LabBin.classes_))

#saving the model
model.save("mask_detector.model", save_format="h5")

model.summary()

model.evaluate(testX, testY)

#plot heatmap
y = np.argmax(testY, axis=1)
sns.heatmap(confusion_matrix(predM, y), annot=True, cmap='rainbow')
plt.ylabel('Predicted output')
plt.xlabel('True output')
plt.savefig("heatmap.png")

#plot the model
N = Epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")




