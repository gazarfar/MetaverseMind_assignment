import os
#--------------------------------------initialization----------------------------------------

import numpy as np
import math as mt
import matplotlib.pyplot as plt
import tensorflow as tf

#Address to datafiles
path2train = 'G:\\New folder\\cv_assignment\\train\\'
path2test = 'G:\\New folder\\cv_assignment\\test\\'


#preprocssing normalizing all the input data to values between 0 and 1
def normalize(image_label, IMG_SIZE = 240):
    image, label = image_label
    image, label = resize_and_rescale(image, label)
    return image, label


#prezise and rescale all data so that they are all 240x240 images and are scaled between 0 and 1
def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [240, 240])
    image = (image / 255.0)
    return image, label

#data augmentation, such as flipping, saturation and brightness adjustments,
#other types of augmentation such as cropping, rotation, and resizing. The current augmentation methods can be addvanced to random augmentation to diversify the trainig set
def augment(image_label, IMG_SIZE = 240):
    image, label = image_label
    image, label = resize_and_rescale(image, label)
    # Random brightness.
    image = tf.image.flip_left_right(image)
    # Random flip
    image = tf.image.adjust_saturation(image, 3)
    # Random brightness
    image = tf.image.adjust_brightness(image, 0.4)
    return image, label


#This function reads the image files and preprocesses the input data
def read_file(path2data,batch_size, valid_spit, Subset):
    if Subset == "test":
        raw_datasets = tf.keras.utils.image_dataset_from_directory(path2data,labels = "inferred",label_mode = "categorical",class_names = ["adidas","converse","nike"], 
                                                                  batch_size=batch_size)
    else:
        raw_datasets = tf.keras.utils.image_dataset_from_directory(path2data,labels = "inferred",label_mode = "categorical",class_names = ["adidas","converse","nike"], 
                                                                   batch_size=batch_size,shuffle = True,seed=500,validation_split = valid_spit,subset =Subset)
    counter = tf.data.experimental.Counter()
    raw_datasets = tf.data.Dataset.zip((raw_datasets, (counter, counter)))
    raw_datasets = raw_datasets.shuffle(1000)
    dataset = raw_datasets.map(normalize)
    if Subset == "training":
        dataset1 = raw_datasets.map(augment)
        dataset = dataset.concatenate(dataset1)
    return dataset

#Reading data
train_dataset = read_file(path2train,batch_size = 32,valid_spit = 0.1, Subset="training")
validation_dataset = read_file(path2train,batch_size = 32,valid_spit = 0.1, Subset="validation")
test_dataset = read_file(path2test,batch_size = 32,valid_spit = 0.1, Subset="test")

#--------------------------------------Model development----------------------------------------
#DenseNet architechture with pretrained wieghts from Imagenet is chosed as the base model for the classification. 
# DensNet require a fewer parameter compare to other equivalent pretrained networks! Please refere to https://arxiv.org/pdf/1608.06993.pdf for more info 
#The input and output layers of the densenet are not included in the design so that we can easily restructure the design for images with arbitary input shape
# a global average pooling layer (for summarizing each channel) and a droppout layer to avoid overfitting are added to the model 
#A prediction leyer with three neuron are reconstructed. Each neuron here is representative of one class

def My_DenseNet121(input_shape):
   


    base_model = tf.keras.applications.densenet.DenseNet121(include_top=False,weights='imagenet',
                                                                             input_shape=input_shape)
     # freeze the base model by making it non trainable
    base_model.trainable = False

    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape) 
    x = base_model(inputs, training=False) 
    
    
    # use global avg pooling to summarize the info in each channel
    x = tf.keras.layers.GlobalAveragePooling2D()(x) 
    # include dropout with probability of 0.2 to avoid overfitting
    x = tf.keras.layers.Dropout(0.2)(x)
        
    # use a prediction layer with three neuron (for 3 class classifier)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    ### END CODE HERE
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

# Adam aptimizer is used to train the model. Since Adam is the combination of gradient decent and RSMP and it needs less memory compare to gradient decent
# The initial learning rate is set at 0.01 here for large epoch number it can be reduced using a learning rate shedulur
# since it is three class classification CategoricalCrossentropy is chosen as the loss function, and the categorial accuracy is chosen for as the metric for evaluation
#Defining the Model
model = My_DenseNet121((240,240,3))
initial_learning_rate = 0.01
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
 #   initial_learning_rate,
 #   decay_steps=10,
 #   decay_rate=0.1,
  #  staircase=True) 

# Define a CategoricalCrossentropy loss function. 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy','categorical_crossentropy'])

model.summary()
#--------------------------------------Training & Evaluation----------------------------------------
# the model is trained by learning rate of 0.01 for two epochs
history = model.fit(train_dataset, epochs=2,validation_data=validation_dataset)#0.01

#the model is trained for 8 more epochs with learning rate of 0.001, since as the model weight get closer to their optimal value, the learning rate must be reduced so that we can converge to the write value

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate*0.1),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy','categorical_crossentropy'])


history = model.fit(train_dataset,epochs=10,initial_epoch=history.epoch[1],validation_data=validation_dataset)

%matplotlib qt
#plotting the training and validation loss to evaluate bias and variance in the dataset
#Because of the shape of the categidical crossentropy loss, we conclude that the model has high bias, so we unfreeze the outer layers of the model and fine tune the model
plt.figure(1)
plt.subplot(1,2,1)#
plt.plot(np.array(history.history['categorical_crossentropy']))
plt.plot(np.array(history.history['val_categorical_crossentropy']))

plt.ylabel('categorical_crossentropy')
plt.xlabel('epoch')
plt.legend(['training','validation'], loc='upper right')

plt.subplot(1,2,2)
plt.plot(np.array(history.history['categorical_accuracy']))
plt.plot(np.array(history.history['val_categorical_accuracy']))

plt.ylabel('categorical_accuracy')
plt.xlabel('epoch')
plt.legend(['training','validation'], loc='upper right')
plt.show()

#--------------------------------------Fine tuning----------------------------------------

base_model = model.layers[1]
base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

#densenet is a large CNN, and it has quite a lot of parameters. The parameters at the inner most layer for a image classification layer is always the same, so the features from imageNet are direclty usable
# and it is enough to only fine tune the outer layers
#since the model is pretrained with Imagenet, most of the pretrained weights from ImageNet at the inner layer can remain constant, and we can fine tune the outer layers of the model with our dataset 

# Fine-tune from this layer onwards
fine_tune_at = 350

### START CODE HERE

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    
# Define a categorical_crossentropy loss function with a smaller learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate*0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy','categorical_crossentropy'])

model.summary()

history_fine = model.fit(train_dataset,epochs=15,initial_epoch=history.epoch[8],validation_data=validation_dataset)

#--------------------------------------test----------------------------------------
#extract the labels from test set and  the predicted labels  for the test set using the trained model

#Labels from test set
test_size = sum(1 for _ in test_dataset)
testlabels = []
for images, labels in test_dataset.take(test_size):
    testlabels = np.append(testlabels,labels.numpy())

testlabels = np.reshape(testlabels,[int(len(testlabels)/3),3])    
testlabels = np.argmax(testlabels, axis = 1)


#Predicting labels from the test set
test_predictions = model.predict(test_dataset)
prediction = np.argmax(test_predictions, axis = 1)

plt.figure(2)
plt.subplot(1,2,1)#
plt.plot(np.array(history_fine.history['categorical_crossentropy']))
plt.plot(np.array(history_fine.history['val_categorical_crossentropy']))

plt.ylabel('categorical_crossentropy')
plt.xlabel('epoch')
plt.legend(['training','validation'], loc='upper right')

plt.subplot(1,2,2)
plt.plot(np.array(history_fine.history['categorical_accuracy']))
plt.plot(np.array(history_fine.history['val_categorical_accuracy']))

plt.ylabel('categorical_accuracy')
plt.xlabel('epoch')
plt.legend(['training','validation'], loc='upper right')
plt.show()

conf_mat = tf.math.confusion_matrix(testlabels,prediction).numpy()
conf_mat = conf_mat/np.sum(conf_mat, axis = 1)*100
print(conf_mat)

# the model does not perform well on training, yet, and it can be improved
#To better the performance, because the model has high bias I will unfreeze more layers, and first make sure that the model has a good preformance on the training set, and then I will evaluate it on the test set!
#by calcualting the reciever operating curve and the area under the ROCs