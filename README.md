# ALTEGRAD Challenge: H Index Prediction
**Team BMV :**

* Bertrand PATUREL
* Martin BRICENO 
* Vanessa CHAHWAN 

<p align="center">
  <img src="https://www.smartdatacollective.com/wp-content/uploads/2018/03/Natural-Language-Processing-NLP-AI-780x445.jpg.webp" width="400" title="NLP">
</p>

# Description
The treatment of data structures such as graphs and texts is a long and complex process considering that there are several methods to extract features from them and that it is not easy to know which of these are suitable for a subsequent task such as classification or regression.
In this repository we will show our methods to extract features  made  in  our  H-index  prediction  process. We will also show the rest of the prediction chain such as regression and post-processing. Finally, this treatment allowed us to be in 2nd place on the public and private leaderboard with a difference of 0.011 in score from the first place in the private leaderboard.

# Project's goal

H-index  is  an  indicator  suggested  in  2005  by  Jorge  Hirsch with the objective of characterizing the quality, relevance and quantifying the scientific production of a researcher. 
This indicator depends on two parameters:
- The number of articles published by the researcher
- The number of citations of each article

A scientist has "index h if h of his or her Np papers have at least h citations each and the other (Np-h) papers have h citations each"

The main purprose of this project is to predict the h-index given:
- A graph indicating which authors have published a paper together.
- The abstracts of several papers in a database.

# Main Model
<p align="center">
  <img src="./main-model.png">
</p>


# Getting started
## Requirements

To install the requirements, run at the root folder:

> ```pip install -r requirements.txt```

## Required files

Before running the model it is advisable to download some files necessary for its execution. This will save you time since the creation of networks and the extraction of some features is very slow. To download the data, you can run at the root folder:

> ```python get_data.py --precomputed PRECOMPUTED```
 
with PRECOMPUTED taking the value 0 if you want to execute all the scripts from scratch and 1 if you want to download the precomputed files.

Or you can manually download the precomputed files using the following download link:

> www.kaggle.com/dataset/def482c59cd5a8ec65e48fef7a67002640d742dfd0e8a389b86421e28356171a

Once downloaded, deposit these files in the ```data``` folder.

## Run our model

To execute our model you have to run the following line in the ```code``` folder.

> ```python model.py```

<p align="center">
  <img src="https://grandes-ecoles.studyrama.com/sites/default/files/styles/content/public/institut-polytechnique-de-paris.jpeg?itok=_Puxulb6" width="300" title="altegrad challenge">
</p>
