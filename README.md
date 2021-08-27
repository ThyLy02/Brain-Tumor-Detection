**[Brain Tumor Detection Project]()**

**[Introduction]()**

This is a project in order to classify tumor or not from brain images.

The model is created with fuctional API.




**[Installation]()**

Create virtual environment
```bash
conda create -n env python=3.8.3
conda activate env
```

Install dependencies
```bash
pip install -r requirements.txt
```

Download and set up data by running
```bash
bash setup_data.sh
```

**[Usage]()**

Run
```bash
python train.py
```

Expected output:


After the models are trained, run:
```bash
python test.py
```



