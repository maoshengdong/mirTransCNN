# mirTransCNN
**Detection and classification of pre-miRNA based on deep learning**

**Training**

- cross-validation for human datasets

python cv_human.py

- cross-validation for cross-species data sets

python cv_whole.py

- testing on human datasets

python main_human.py



**Usage**

python isMiRNA.py  -s <RNAsequence\> -m <model_path\> -f <feature\>

for example: python isMiRNA.py -s AGAAUUCUCUUAUCCAACAUCAACAUCUUGGUCAGAUUUGAACUCUUCAA -m model/human/model_cv_human_3.pkl -f feature_data/featureSelectedModel_cv_human_3.pkl

python isMiRNA.py -i \<input file\> -o \<output file\>

- input file: fasta format

- output file: txt format

- model_path: the path of trained model

- feature：Algorithm model for feature extraction

  

**Dependencies**

1. Python >= 3.6
2. RNAFold
4. Pytorch
4. Numpy



**File**

1. data.zip
- Datasets containing humans and cross-species

2. feature_data.zip
- The sequence feature set used in this experimental method, File names with "whole" represent cross-species, and those with "human" represent human.

3. model
- The path of trained model



