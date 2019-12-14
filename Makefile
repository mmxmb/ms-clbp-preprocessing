
install:
	pip3 install numpy>=1.14.1 scipy>=0.19.0 matplotlib>=2.0.0,!=3.0.0 networkx>=2.0 pillow>=4.3.0 imageio>=2.3.0 PyWavelets>=0.4.0 Cython
	pip3 install git+https://github.com/mmxmb/scikit-image
	pip3 install -r requirements.txt

run:
	python3 ms_clbp_preprocessing/preprocessing.py

.PHONY: install run
