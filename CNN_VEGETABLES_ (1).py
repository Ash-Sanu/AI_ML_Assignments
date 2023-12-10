#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras 
from tensorflow.keras import layers


# In[9]:


data_train_path = 'C:/Users/aswin/OneDrive/Desktop/CNN Vegetables/Vegetables/Vegetable Images/train'    # Train Dataset
data_test_path = 'C:/Users/aswin/OneDrive/Desktop/CNN Vegetables/Vegetables/Vegetable Images/test'     # Test dataset
data_val_path = 'C:/Users/aswin/OneDrive/Desktop/CNN Vegetables/Vegetables/Vegetable Images/validation'  # validation dataset


# In[10]:


img_width  =180
img_height = 180


# In[11]:


# Train dataset


# In[12]:


data_train = tf.keras.utils.image_dataset_from_directory(
data_train_path,
shuffle = True,
image_size=(img_width,img_height),
batch_size=32,
validation_split=False)


# In[13]:


data_cat = data_train.class_names


# In[14]:


# Validation Dataset


# In[15]:


data_val = tf.keras.utils.image_dataset_from_directory(data_val_path,
                                                      image_size=(img_height,img_width),
                                                      batch_size=32,
                                                      shuffle=False,
                                                      validation_split=False)


# In[16]:


# Test dataset


# In[17]:


data_test = tf.keras.utils.image_dataset_from_directory(data_test_path,
                                                      image_size=(img_height,img_width),
                                                      batch_size=32,
                                                      shuffle=False,
                                                      validation_split=False)


# In[20]:


# Viewing the Train dataset
plt.figure(figsize=(10,10))
for image,labels in data_train.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(data_cat[labels[i]])
        plt.axis('off')


# # Building the Model

# In[22]:


from tensorflow.keras.models import Sequential


# In[26]:


data_train


# In[ ]:


# Model parameters


# In[29]:


model = Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128),
    layers.Dense(units = len(data_cat))
])


# In[30]:


model.compile(optimizer='adam', loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


# In[ ]:


# Fitting model


# In[32]:


epochs = 15
history = model.fit(data_train, validation_data=data_val, epochs=epochs, batch_size=32, verbose=1)


# In[ ]:





# In[34]:


epochs_range = range(epochs)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, history.history['accuracy'], label = 'Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, history.history['loss'], label = 'Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label = 'Validation Loss')
plt.title('Loss')


# In[35]:


# Testing if the model works


# In[37]:


image = 'C:/Users/aswin/OneDrive/Desktop/CNN Vegetables/Vegetables/Vegetable Images/validation/Cauliflower/1256.jpg'
image = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat = tf.expand_dims(img_arr,0)


# In[38]:


predict = model.predict(img_bat)


# In[39]:


# Predicting Score
score = tf.nn.softmax(predict)


# In[42]:


print('Vegetable predicted is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)], np.max(score)*100))


# In[43]:


image = 'C:/Users/aswin/OneDrive/Desktop/CNN Vegetables/Vegetables/Vegetable Images/validation/Broccoli/1216.jpg'
image = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat = tf.expand_dims(img_arr,0)


# In[44]:


predict = model.predict(img_bat)


# In[45]:


# Predicting Score
score = tf.nn.softmax(predict)


# In[46]:


print('Vegetable predicted is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)], np.max(score)*100))


# In[49]:


model.save('Image_classify.keras')


# # Yet Another approach could be

# # 1. Install Dependencies and Setup

# In[47]:


get_ipython().system('pip install tensorflow')


# In[48]:


get_ipython().system('pip install tensorflow tensorflow-gpu opencv-python matplotlib')


# In[50]:


get_ipython().system('pip list')


# In[51]:


import tensorflow as tf
import os


# # 2. Remove dodgy images for train and test dataset

# In[52]:


get_ipython().system('pip install opencv-python')


# In[53]:


import cv2
import imghdr


# In[54]:


data_dir_train = 'C:/Users/aswin/OneDrive/Desktop/CNN Vegetables/Vegetables/Vegetable Images/train'    # Train Dataset
data_dir_test = 'C:/Users/aswin/OneDrive/Desktop/CNN Vegetables/Vegetables/Vegetable Images/test'     # Test dataset
data_dir_val = 'C:/Users/aswin/OneDrive/Desktop/CNN Vegetables/Vegetables/Vegetable Images/validation'  # validation dataset


# In[56]:


image_exts = ['jpeg','jpg', 'bmp', 'png']


# In[57]:


# Removing dodgy images for train dataset
for image_class in os.listdir(data_dir_train): 
    for image in os.listdir(os.path.join(data_dir_train, image_class)):
        image_path = os.path.join(data_dir_train, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)
            
# Removing dodgy images for test dataset            
for image_class in os.listdir(data_dir_test): 
    for image in os.listdir(os.path.join(data_dir_test, image_class)):
        image_path = os.path.join(data_dir_test, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)
            
# Removing dodgy images for validate dataset            
for image_class in os.listdir(data_dir_val): 
    for image in os.listdir(os.path.join(data_dir_val, image_class)):
        image_path = os.path.join(data_dir_val, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)


# # 3. Load Data

# In[310]:


img_height = 256
img_width = 256


# In[311]:


import numpy as np
from matplotlib import pyplot as plt


# In[312]:


# train dataset


# In[313]:


data = tf.keras.utils.image_dataset_from_directory(
data_dir_train,
shuffle = True,
image_size=(img_width,img_height),
batch_size=32,
validation_split=False)


# In[314]:


data_iterator = data.as_numpy_iterator()


# In[315]:


batch_train = data_iterator.next()


# In[316]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch_train[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch_train[1][idx])


# In[317]:


# test dataset


# In[318]:


data1 = tf.keras.utils.image_dataset_from_directory(
data_dir_test,
shuffle = True,
image_size=(img_width,img_height),
batch_size=32,
validation_split=False)


# In[319]:


data_iterator = data1.as_numpy_iterator()


# In[320]:


batch_test = data_iterator.next()


# In[321]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch_test[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch_test[1][idx])


# In[322]:


# Validation dataset


# In[323]:


data2 = tf.keras.utils.image_dataset_from_directory(
data_dir_test,
shuffle = True,
image_size=(img_width,img_height),
batch_size=32,
validation_split=False)


# In[324]:


data_iterator = data2.as_numpy_iterator()


# In[325]:


batch_val = data_iterator.next()


# In[326]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch_val[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch_val[1][idx])


# # Pre-processing Data

# # 4. Scale Data

# In[327]:


# train dataset


# In[328]:


data = data.map(lambda x,y: (x/255, y))


# In[329]:


scaled_iterator = data.as_numpy_iterator()


# In[330]:


batch_train = scaled_iterator.next()


# In[331]:


batch_train[0].max()


# In[332]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch_train[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch_train[1][idx])


# In[333]:


# test dataset


# In[334]:


data1 = data1.map(lambda x,y: (x/255, y))


# In[335]:


scaled_iterator = data1.as_numpy_iterator()


# In[336]:


batch_test = scaled_iterator.next()


# In[337]:


batch_test[0].max()


# In[338]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch_test[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch_test[1][idx])


# In[339]:


# Validation dataset


# In[340]:


data2 = data2.map(lambda x,y: (x/255, y))


# In[341]:


scaled_iterator = data2.as_numpy_iterator()


# In[342]:


batch_val = scaled_iterator.next()


# In[343]:


batch_val[0].max()


# In[344]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch_val[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch_val[1][idx])


# # 5. Splitting the data

# In[345]:


print(len(data),len(data1),len(data2))


# In[346]:


train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)


# In[347]:


train_size


# In[348]:


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


# # 6. Build Deep Learning Model

# In[349]:


train


# In[350]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[351]:


model = Sequential()   # Creating an instance


# In[352]:


# Model Paramters
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))                            
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[353]:


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


# In[354]:


model.summary()


# # 7. Train

# In[355]:


hist = model.fit(train, epochs=15, batch_size=32,validation_data=val)


# # 8. Plot Performance

# In[ ]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[ ]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# # 9. Evaluate

# In[ ]:


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[ ]:


pre = Precision()
re = Recall()
acc = BinaryAccuracy()


# In[ ]:


for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


# In[ ]:


print(pre.result(), re.result(), acc.result())


# # 10. Test

# In[ ]:


import cv2


# In[ ]:


# For Cauliflower
image = 'C:/Users/aswin/OneDrive/Desktop/CNN Vegetables/Vegetables/Vegetable Images/validation/Cauliflower/1256.jpg'
image = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat = tf.expand_dims(img_arr,0)


# In[ ]:


predict = model.predict(img_bat)


# In[ ]:


# Predicting Score
score = tf.nn.softmax(predict)


# In[ ]:


print('Vegetable predicted is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)], np.max(score)*100))


# In[ ]:


# For Brocolli
image = 'C:/Users/aswin/OneDrive/Desktop/CNN Vegetables/Vegetables/Vegetable Images/validation/Broccoli/1216.jpg'
image = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat = tf.expand_dims(img_arr,0)


# In[ ]:


predict = model.predict(img_bat)


# In[ ]:


# Predicting Score
score = tf.nn.softmax(predict)


# In[ ]:


print('Vegetable predicted is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)], np.max(score)*100))


# # 11. Save the Model

# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


model.save(os.path.join('models','imageclassifier.h5'))


# In[ ]:


new_model = load_model('imageclassifier.h5')


# In[ ]:


new_model.predict(np.expand_dims(resize/255, 0))


# In[ ]:




