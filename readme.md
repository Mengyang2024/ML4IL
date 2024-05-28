# ML4IL: Predictive machine learning models for ionic liquids 

ML4IL is a repository about predictive machine learning models for ionic liquids. Currently, models for predicting cellulose solubility and melting point of ionic liquids have been developed.

## Usage

### Installation:
Clone the repository and install the required packages:
```shell
conda create -n ML4IL python=3.8
conda activate ML4IL
conda install cudatoolkit=11.8.0
conda install cudnn=8.9.2.26 -c anaconda
pip install tensorflow==2.13.0 rdkit==2023.9.4 pandas==1.5.3 scikit-learn==1.3.0 shap==0.44.1 matplotlib
```

### ML_for_cellulose_solubility_prediction
Experimental data and code for developing machine learning models to predict cellulose solubility in ionic liquids.

### ML_for_melting_point_prediction
Experimental data and code for developing machine learning models predict melting points of ionic liquids.

## Reference

If you find the code useful for your research, please consider citing

```
@inproceedings{Mengyang_2024_IonGen,
  author       = {Mengyang Qu, Gyanendra Sharma, Naoki Wada, Hisaki Ikebata, Shigeyuki Matsunami and Kenji Takahashi},
  title        = {Machine Learning-Driven Generation and Screening of Potential Ionic Liquids for Cellulose Dissolution},
  year         = 2024,
  publisher    = {},
  howpublished = {},
}
```