from src.model import BrainTumorModel
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


import os
from PIL import Image
import numpy as np
import cv2


# load image
encoder = OneHotEncoder()
encoder.fit([[0], [1]])

data = []
paths = []
result = []


for r, d, f in os.walk(r'./data/brain-tumor-dataset/brain-tumor-dataset/yes'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128, 128))
    img = np.array(img)

    if(img.shape == (128, 128, 3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())
result[0]


paths = []

for r, d, f in os.walk(r'./data/brain-tumor-dataset/brain-tumor-dataset/no'):
     for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128, 128))
    img = np.array(img)

    if(img.shape == (128, 128, 3)):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray())


data = np.array(data)

result = np.array(result)
result = result.reshape(139, 2)


# train-test-split
train_images, test_images, train_labels, test_labels = train_test_split(data, result,
                                                                        test_size=0.1, shuffle=True, random_state=1)


# load model
model = BrainTumorModel().load_model()


def names(number):
    if number == 0:
        return 'it is a Tumor'
    else:
        return 'No, it is not a tumor'

# test image



img = Image.open(r'./data/brain-tumor-dataset/brain-tumor-dataset/yes/Y112.jpg')


x = np.array(img.resize((128, 128)))
x = x.reshape(1, 128, 128, 3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
print(str(res[0][classification]*100) + '% Confidence ' + names(classification))
img = cv2.imread(r'./data/brain-tumor-dataset/brain-tumor-dataset/yes/Y112.JPG', 0)
cv2.imshow('Display image', img)
cv2.waitKey(7000)



