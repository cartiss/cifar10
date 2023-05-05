from tensorflow import keras
import numpy as np
import cv2

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
new_model = keras.models.load_model('models/cnn_cifar10.h5')

cat_image = np.array(cv2.resize(cv2.imread('test_images/horse.jpg'), (32, 32)))
print(cat_image.shape)
predict = new_model.predict(np.array([cat_image]))
print(class_names[np.argmax(predict)])