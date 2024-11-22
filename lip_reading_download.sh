#!/bin/bash

# Download external library
cd src/datasets/lip_dataset
git clone --recursive https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks.git

# Rename
mkdir lipreading
mv Lipreading_using_Temporal_Convolutional_Networks/* lipreading
rm -rf Lipreading_using_Temporal_Convolutional_Networks

# Installing requirements
cd lipreading
pip install -r requirements.txt

# Adding __init__.py files for used modules
touch __init__.py lipreading/__init__.py lipreading/models/__init__.py

# Rename imports
rename() {
    sed -i 's/from lipreading./from src.datasets.lip_dataset.lipreading.lipreading./g' "$1"
}

rename "main.py"
for file in lipreading/*.py; do
    rename "$file"
done

for file in lipreading/models/*.py; do
    rename "$file"
done
