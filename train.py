from src.model import BrainTumorModel
from src.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


import os
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('dark_background')

# Load data
encoder = OneHotEncoder()
encoder.fit([[0], [1]])


data = []
paths = []
result = []

<<<<<<< HEAD
for r, d, f in os.walk(r'./data/brain-tumor-dataset/brain-tumor-dataset/yes'):
=======
for r, d, f in os.walk(r'./data/brain-tumor-dataset/yes'):
>>>>>>> d92cd5bd13ecc40cec77e1020eac996d2d07439c
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128, 128))
    img = np.array(img)

    if (img.shape == (128, 128, 3)) :
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())
result[0]


paths = []
<<<<<<< HEAD
for r, d, f in os.walk(r'./data/brain-tumor-dataset/brain-tumor-dataset/no'):
=======
for r, d, f in os.walk(r'./data/brain-tumor-dataset/no'):
>>>>>>> d92cd5bd13ecc40cec77e1020eac996d2d07439c
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

# Train-Test split
train_images, test_images, train_labels, test_labels = train_test_split(data, result,
                                                                        test_size=0.1, shuffle=True, random_state=1)

# load model
model = BrainTumorModel().load_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# train
history = model.fit(train_images, train_labels, epochs=40,
                    batch_size=128, validation_data=(test_images, test_labels))


# summary model
model = BrainTumorModel().summary_model()

# plot model
plot_model(history)
