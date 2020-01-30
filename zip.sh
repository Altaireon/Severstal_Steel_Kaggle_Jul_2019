#!/bin/bash
rm -rf model.zip
conda env export > environment.yaml 
zip -r model.zip notebooks external_libs deep_learning params data/lib data/loader.py data/lenna.bmp outputs runner environment.yaml external_libs -x='outputs/*-*'
