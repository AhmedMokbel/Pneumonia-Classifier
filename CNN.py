#Convolutional Neural Network "CNN"
import  matplotlib.pyplot as plt
#Part"1" :Building CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D ,SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense ,Dropout ,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint , ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import  ImageDataGenerator

batch_size=32
epochs=10
height=150
width=150

#initializing CNN
model=Sequential()

"""
#Step-1 :Convolution
model.add(Convolution2D(filters=16,kernel_size=(3,3),input_shape=(height,width,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Add Second Convolutional layer
model.add(Convolution2D(filters=32,kernel_size=(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) 

#Add three Convolutional layer
model.add(Convolution2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) 

#Add four Convolutional layer
model.add(Convolution2D(filters=128,kernel_size=(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) 
"""
# First conv block
model.add(Convolution2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same' ,input_shape=(height,width,3)))
model.add(Convolution2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

# Second conv block
model.add(SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))


# third conv block
model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))


# Fourth conv block
model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

# Fifth conv block
model.add(SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))


#Step-3 : Flattening
model.add(Flatten())

#Step-4:Full_Connection layer
model.add( Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.7))
model.add( Dense(units=128, activation='relu'))
model.add(Dropout(rate=0.5))
model.add( Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.3))

#output layer
model.add(Dense(units=1,activation="sigmoid"))

#compile CNN
model.compile(optimizer="adam" ,loss="binary_crossentropy",metrics=["accuracy"])



#save model
filepath="model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
callbacks_list = [checkpoint ,lr_reduce] 

#Part"2" :Fitting CNN to image

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)


train_set = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(height, width),
        batch_size=batch_size ,
        class_mode='binary')



test_set = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='binary')

valid_test = valid_datagen.flow_from_directory(
        'dataset/val',
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='binary')


history=model.fit(train_set,
                         steps_per_epoch = train_set.samples // batch_size,
                         epochs = epochs,
                          verbose=1,
                         validation_data = test_set,
                         callbacks=callbacks_list ,
                          validation_steps = test_set.samples // batch_size)





# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




