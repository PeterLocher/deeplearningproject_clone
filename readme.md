**Step 1** - install packages:

For training and experiments:
*keras,
keras-unet,
numpy,
matplotlib,
pillow*

(For preprocessing:
*cv2*)

**Step 2** - download data and models (the first 2 are sufficient to run the experiments):

Rural China 256x256 (4GB): *https://mega.nz/folder/cPlAhA4C#sHA7wvc5w8rOMyLtYf5APQ*

Trained models (1.7GB):**

Rural China 1024x1024 (4GB):**

Poland 1024x1024 (1.8GB):**

**Step 3** - update constants.py:
Edit the *china_path* and *poland_path* variables in the *constants.py* file to point to the outermost folders of the data.


**Step 4** - run:

Run *training.py* to train new modes and *experiments.py* to show experiments
