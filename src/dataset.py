from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import os
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('dark_background')


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

print(result[0])
print(result.shape)
print(data.shape)

# Train-Test split
train_images, test_images, train_labels, test_labels = train_test_split(data, result,
                                                                        test_size=0.1, shuffle=True, random_state=1)
print('Train Images Shape', train_images.shape)
print('Train Labels Shape', train_labels.shape)
print('Test Images Shape', test_images.shape)
print('Test Labels Shape', test_labels.shape)