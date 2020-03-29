#Convolutional Neural Network "CNN"

#Part"1" :Building CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


#initializing CNN
classifier=Sequential()

#Step-1 :Convolution
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),input_shape=(64,64,3),activation="relu"))

#Step-2 : Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Add Second Convolutional layer
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step-3 : Flattening
classifier.add(Flatten())

#Step-4:Full_Connection
classifier.add(Dense(256, activation='relu'))
classifier.add(Dense(50, activation='relu'))
classifier.add(Dense(units=2,activation="softmax"))

#compile CNN
classifier.compile(optimizer="adam" ,loss="categorical_crossentropy",metrics=["accuracy"])

#Part"2" :Fitting CNN to image
from keras.preprocessing.image import  ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(64, 64),
        batch_size=1 ,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(64, 64),
        batch_size=1,
        class_mode='categorical')

valid_test = valid_datagen.flow_from_directory(
        'dataset/val',
        target_size=(64, 64),
        batch_size=1,
        class_mode='categorical')


history=classifier.fit_generator(training_set,
                         steps_per_epoch = 5126,
                         epochs = 16,
                         validation_data = valid_test,
                          validation_steps = 16)



#Evaluate model on test set
score = classifier.evaluate_generator(test_set,workers=3)

test_set.reset() #Necessary to force it to start from beginning
scores= classifier.predict_generator(test_set,workers=3)


NORMAL = 0
PNEUMONIA=0
for i, n in enumerate(test_set.filenames):
    if n.startswith("NORMAL") and scores[i][0] >= 0.5:
        NORMAL += 1
    if n.startswith("PNEUMONIA") and scores[i][1] >= 0.5:
        PNEUMONIA += 1

print("NORMAL_CORRECT:",NORMAL,"PNEUMONIA_CORRECT:", PNEUMONIA, " Total: ", len(test_set.filenames))
print("Loss: ", score[0], "Accuracy: ", score[1])



