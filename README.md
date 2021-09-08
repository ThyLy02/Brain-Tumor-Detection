# **Brain Tumor Detection Project**

## **Introduction**

This is a project in order to classify tumor or not from brain images.

The model is created with functional API.

The brain-tumor-dataset are collected from Kaggle. There are 253 files image included 155 file yes (tumor) and 98 files no (no tumor).

![no tumor](https://github.com/ThyLy02/Brain-Tumor-Detection/blob/main/images/brain_image_no.png)

![tumor](https://github.com/ThyLy02/Brain-Tumor-Detection/blob/main/images/brain_image_yes.png)


## **Acknowledgements**

Organizing project: https://github.com/qnn122/organizing-training-project-tutorial

Data source: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection/

## **Installation**

### Create virtual environment:
```bash
conda create -n brainenv python=3.8
conda activate brainenv
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Download and set up data by running:
```bash
bash setup_data.sh
```

## **Usage**

### Run:
```bash
python train.py
```

### Expected output:
```bash
Epoch 1/40
1/1 [==============================] - 49s 49s/step - loss: 1.6247 - accuracy: 0.4640 - val_loss: 680.0847 - val_accuracy: 0.3571
Epoch 2/40
1/1 [==============================] - 113s 113s/step - loss: 80.7473 - accuracy: 0.5520 - val_loss: 73.8134 - val_accuracy: 0.7143
```

## **Prediction**

### After the models are trained, run:
```bash
python test.py
```



