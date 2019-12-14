
install:
	pip install -r requirements.txt --no-index --find-links file://ms_clbp_preprocessing/scikit-image

run:
	python3 ms_clbp_preprocessing/preprocessing.py

.PHONY: install run
