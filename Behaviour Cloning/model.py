import csv
import numpy as np
from time import time
from sklearn.utils import shuffle
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D,Cropping2D

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    

def data_create(samples):
    
    num_samples = len(samples)
    
    shuffle(samples)
    t0=time()
    
    car_images=[]
    steering_angles=[]
    
    augmented_images = []
    augmented_measurement = []

    t0=time()

    for batch_sample in samples:

        car_images = []
        steering_angles = []

        steering_center=float(batch_sample[3])
        
        path = "data/IMG/" # fill in the path to your training IMG directory
        img_center = cv2.imread(path + batch_sample[0].split('/')[-1])
    
        car_images.extend([img_center])
        steering_angles.extend([steering_center])

    for car_image, steering_angle in zip(car_images, steering_angles):
        augmented_images.append(car_image)
        augmented_measurement.append(steering_angle)

        augmented_images.append(cv2.flip(car_image, 1))
        augmented_measurement.append(-1.0 * steering_angle)

    del car_images,steering_angles

    # trim image to only see section with road
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurement)
    print('time spend on one batch:{}'.format(time()-t0))
    
    del augmented_images,augmented_measurement
    
    return X_train,y_train


X_train,y_train=data_create(lines)

model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))

model.add(Conv2D(kernel_size=5, strides=1, filters=24, padding='SAME', activation='relu'))
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='SAME', activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=5, strides=1, filters=48, padding='SAME', activation='relu'))
model.add(Conv2D(kernel_size=5, strides=1, filters=64, padding='SAME', activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=5, strides=1, filters=128, padding='SAME', activation='relu'))
model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object=model.fit(x=X_train,y=y_train,batch_size=128,epochs=1,verbose=1,validation_split=0.2)	

model.save('model.h5')


print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
