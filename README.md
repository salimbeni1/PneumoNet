# PneumoNet: Neural networks for the detection of pneumonia from digital lung auscultation audio


Precise lung sounds classification is still an open issue: traditional auscultation methods are
limited due to biased human interpretation. This paper introduce two CNN models, which differ in the way
audio crops are combined and fed to the network. They recognize specific audio patterns in the STFT
spectrograms and classify healthy and unhealthy pediatric patients, suffering from different pulmonary
diseases. Two different datasets are investigated and compared rigorously; however, the attempts to find a
general model that performed well on both sets were not successful. After a weighted mean aggregation method,
both models achieved an accuracy of 94%.

see report : [PneumoNet](https://github.com/CS-433/cs-433-project-2-deepnettonepanettone/blob/main/PneumoNet.pdf)

## Environment setup

 You can use the notebooks directly on a local directory or on google colab in order to use their free GPUs.

In both scenario, please :
```
git clone https://github.com/CS-433/cs-433-project-2-deepnettonepanettone
cd cs-433-project-2-deepnettonepanettone/src
```

at the source of the repo add a shortcut using the link or copy the content of this google drive folder (preporcessed dataset) :
```
https://drive.google.com/drive/folders/1akCqrEMU8lgYp0bHNEoPEUQdTjKwGur2?usp=sharing
```

## Dependencies

make sure the python environment has : 

* numpy
* scipy
* librosa
* maplotlib
* tensorflow - keras

only if you want to test data augentation technics
* nlpaug

only if you want to test Grad-Cam class activation 
* tf-keras-vis

use pip or conda install


>
> Warining : the processed dataset is huge more than 30 GB
> All the following run can take up to 40 min just in data processing
> Make sure you have 12 GB of RAM available, and not to upload more than 2 batch 
> on the same runtime, especially on the model by patient as it using early fusion. 
>

## Where to find the best models

in src/model/ you can find the most succeful models

* best model for GVA with the model by Patient : TRAIN_ON_GVA_REM_POS_9/TRAIN_ON_GVA_TFINAL_RUN_3_-2020-12-12.h5
* best model for GVA with the model by Position : MPO_GVA_best/

* best model for POA with the model by Patient : TRAIN_ON_POA_BATCH_1/TRAIN_ON_POA_BATCH_1_RUN_7_-2020-12-12.h5
* best model for POA with the model by Position : MPO_POA_best/

You can also find all the saved models we used for the paper in this model/ folder

## Run the best models

> you can adapt the all notebooks with the model weights you are curious to test

run the notebooks for

* best model for GVA with the model by Patient with : confusion_matrix_MPA.ipynb
* best model for GVA with the model by Position with : MPO_10fold_cross_validation_and_prediction.pynb

* best model for POA with the model by Patient with : confusion_matrix_MPA.ipynb
* best model for POA with the model by Position with : MPO_10fold_cross_validation_and_prediction.ipynb

## Test our validation methods

>
> All the main functions are in utils.py and sound_processing.py
> The notebooks (that use these functions) are here as examples and proof for our results
>

* 10fold cross validation for GVA : MPO_10fold_cross_validation_and_prediction.ipynb
* data augmentation : use sound_processing.augmented function
* Grad-Cam class activation : grad_cam_POA.ipynb
* different types of spectrogram : \*\_MEL.h5 and \*\_MFCC.h5 
* position relevance : Remove_Position_Analysis.ipynb
* transfer learning : 
    * GVA to POA1  \*\_TRANSF\_\*/run2 weights
    * GVA to POA2  \*\_TRANSF\_\*/run6 weights
    * POA1 to GVA  \*\_TRANSF\_\*/run5 weights
    * POA2 to GVA  \*\_TRANSF\_\*/run8 weights
* Monte carlo dropout : MPO_10fold_cross_validation_and_prediction.ipynb
* types of normalisation : \*\_STFT_FEAT/ , \*\_STFT_SAMPLE/

## Authors

* Sara Pagnamenta
* Luka Chinchaladze
* Etienne Salimbeni

## Acknowledgements

Thank you !!!!