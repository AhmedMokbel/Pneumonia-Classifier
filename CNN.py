#Convolutional Neural Network "CNN"
import  matplotlib.pyplot as plt
import  numpy as np
import  cv2
import os
#Part"1" :Building CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense ,Dropout ,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint , ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.models import load_model

batch_size=32
epochs=20
height=150
width=150

#initializing CNN
model=Sequential()


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

#Add fifth Convolutional layer
model.add(Convolution2D(filters=256,kernel_size=(3,3),activation="relu"))
model.add(Convolution2D(filters=512,kernel_size=(3,3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) 





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




#calculate the accuracy of test set and valdation set



classifier=load_model("model.hdf5")

test_data = []
test_labels = []
input_path="dataset/"
for cond in ['/NORMAL/', '/PNEUMONIA/']:
    for img in (os.listdir(input_path + 'test' + cond)):
            img = plt.imread(input_path+'test'+cond+img)
            img = cv2.resize(img, (height, width))
            img = np.dstack([img, img, img])
            img = img.astype('float32') / 255
            if cond=='/NORMAL/':
                label = 0
            elif cond=='/PNEUMONIA/':
                label = 1
            test_data.append(img)
            test_labels.append(label)
        
test_data = np.array(test_data)
test_labels = np.array(test_labels) 

from sklearn.metrics import accuracy_score, confusion_matrix
preds = classifier.predict(test_data)


acc = accuracy_score(test_labels, np.round(preds))*100
cm = confusion_matrix(test_labels, np.round(preds))
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX ------------------')
print(cm)

print('\nTEST METRICS ----------------------')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2*precision*recall/(precision+recall)))




