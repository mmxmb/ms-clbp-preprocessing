# CNN preprocessing using MS-CLBP

This repository contains source code for the paper: [Convolutional neural network preprocessing usingmulti-scale completed local binary patterns - Mikhaylov, 2019](https://www.mmxmb.net/files/ms_clbp_preprocessing.pdf)

## Description

The goal of this project was to attempt to extract useful features from images using multi-scale completed local binary patterns (MS-CLBP). The effectiveness of the approach is tested by training a CNN on the original dataset and on preprocessed dataset, and comparing results.

## Project Structure

```
├── Makefile
├── README.md
├── classification                  # Jupyter notebooks containing CNN training and evaluation code
│   ├── ms_clbp_cnn.ipynb           # MS-CLBP + CNN
│   └── no_preprocessing_cnn.ipynb  # no preprocessing CNN
├── dataset                         # default location of image dataset
├── misc                            # contains Jupyter notebook demonstrating how MS-CLBP method works
│   ├── freeway65.tif
│   └── lbp_demo.ipynb
├── ms_clbp_preprocessing
│   ├── ms_clbp.py                  # MS-CLBP method code
│   ├── preprocessing.py            # run this file to preprocess the dataset
│   └── scikit-image                # forked scikit-image
├── pkl                             # default location for writing pickled MS-CLBP feature tensor during preprocessing
└── requirements.txt
```

## Using this code

### Installation

This project uses a fork of [scikit-image](https://github.com/mmxmb/scikit-image) package that contains several additional feature extraction Cython functions. This project also has lots of dependencies because both deep learning and traditional computer vision techniques are used throughout the project.

The easiest way to install all dependencies, including forked `scikit-image` is to run:

```
make install
```

### Preprocessing

You can preprocess a dataset using MS-CLBP method using the following command:

```
make run
```

This command in turn calls `ms_clbp_preprocessing/preprocessing.py`. Examine that file and set the appropriate values for the following variables:
* `scales`
* `n_points`
* `radii`
* `patch_size`
* `n_bins`
* `classes`
* `dataset_path`
* `target_path`

By default, these values are set to what was used in the experiment. The only thing required to make this code work with default settings is to load the dataset into `dataset` directory; one directory per class. _UC Merced Land Use Dataset_ is available [here](UC Merced Land Use Dataset). 

### Training and evaluation

CNN training and evaluation code is located in `classification` directory in the form of Jupyter notebooks.

## Acknowledgements

[1] T.Ojala,M.Pietikainen,andT.Maenpaa,“Multiresolutiongray-scaleand rotation invariant texture classification with local binary patterns,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, no. 7, pp. 971–987, July 2002.
[2] C. Chen, W. Li, and Q. Du, “Remote sensing image scene classification using multi-scale completed local binary patterns and fisher vectors,” Remote Sensing, vol. 8, p. 483, 06 2016.
[3] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller, “Striv- ing for Simplicity: The All Convolutional Net,” arXiv e-prints, p. arXiv:1412.6806, Dec 2014.
[4] F. Juefei-Xu, V. Naresh Boddeti, and M. Savvides, “Local Binary Con- volutional Neural Networks,” arXiv e-prints, p. arXiv:1608.06049, Aug 2016.

## License

MIT License

Copyright (c) [2019] [Maxim Mikhaylov]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
