#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import os, glob

print("PIL 라이브러리 import 완료!")


# In[2]:


import os

def resize_images(img_path):
        images=glob.glob(img_path + "/*.jpg")  
    
        print(len(images), " images to be resized.")

        target_size=(28,28)
        for img in images:
            old_img=Image.open(img)
            new_img=old_img.resize(target_size,Image.ANTIALIAS)
            new_img.save(img, "JPEG")
    
        print(len(images), " images resized.")

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/scissor"
resize_images(image_dir_path)

print("가위 이미지 resize 완료!")


# In[3]:


image_dir_path_paper = os.getenv("HOME") + "/aiffel/rock_scissor_paper/paper"

resize_images(image_dir_path_paper)


# In[4]:


image_dir_path_rock = os.getenv("HOME") + "/aiffel/rock_scissor_paper/rock"

resize_images(image_dir_path_rock)


# In[6]:


import numpy as np

def load_data(img_path, number_of_data=900):

    img_size=28
    color=3
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img 
        labels[idx]=0  
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img 
        labels[idx]=1 
        idx=idx+1  
    
    for file in glob.iglob(img_path+'/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img 
        labels[idx]=2 
        idx=idx+1
        
    print("학습데이터의 이미지 개수는", idx,"입니다.")
    return imgs, labels

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper"
(x_train, y_train)=load_data(image_dir_path)
x_train_norm = x_train/255.0 

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))


# In[7]:


import matplotlib.pyplot as plt
plt.imshow(x_train[0])
print('라벨: ', y_train[0])


# In[8]:


import tensorflow as tf
from tensorflow import keras
import numpy as np

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,3), padding='same'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()


# In[10]:


keras.utils.plot_model(model, show_shapes=True,
                       to_file='cnn-architecture.png', dpi=100)


# In[12]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(x_train_norm, y_train, epochs=10)


# In[13]:


import os

image_dir_path_test_scissor = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/scissor"
resize_images(image_dir_path_test_scissor)

image_dir_path_test_paper = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/paper"
resize_images(image_dir_path_test_paper)

image_dir_path_test_rock = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/rock"
resize_images(image_dir_path_test_rock)

image_dir_path_test = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test"
(x_test, y_test)=load_data(image_dir_path_test)
x_test_norm = x_test/255.0

print("x_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))


# In[14]:


test_loss, test_accuracy = model.evaluate(x_test_norm, y_test, verbose=2)


# In[ ]:




