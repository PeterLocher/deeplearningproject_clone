**Step 1** - install packages for Python 3.8:

For training and experiments:
*keras,
keras-unet,
numpy,
matplotlib,
pillow*

For preprocessing:
*cv2* (not necessary, data is uploaded in preprocessed form)

**Step 2** - download data and models. Open links and press *Download all as zip*, then unwrap into the root folder of the project.  The first 2 links are sufficient to run the sample experiments

Rural China 256x256 (0.5GB): *https://mega.nz/folder/cPlAhA4C#sHA7wvc5w8rOMyLtYf5APQ* (contains only 15% of the original set)

Trained models (0.6GB): *https://mega.nz/file/1fdDAKZB#GUoZmgvSel34-n8T6i5QFatxg0khTslbg-65b0ry4NE* (contains only the models currently used in the scripts ~30% of all models i saved)

Rural China 1024x1024 (0.4GB): *https://mega.nz/folder/kfkhiSJS#HkCUfC5PNys0SZ39Ry55Pw* (contains only 10% of the original set)

Poland 1024x1024 (0.2GB): *https://mega.nz/folder/dCNHRZDB#b9qO48tRhXrYDYO6wHiJBw* (contains only 10% of the original set)

**Step 3** - update constants.py:

Edit the *china_path* and *poland_path* variables in the *constants.py* file to point to the outermost folders of the data.


**Step 4** - run:

Run *training.py* to train new modes and *experiments.py* to show experiments, or run the cells in *notebook.ipynb*.
