# local_global_attention_model
This repository includes the code for the following paper - https://arxiv.org/abs/2004.04649

### Source code
The source code of the model can be found in the src model.
It contains a module for each version of the model. 
The base module is LocalGlobalAttentionModel and it contains the basic functionality of the model. The rest of the modules expend this module.
The main functionalities of the Model is data generation and inference. For examples how to use these functionalities please see the Experiments and notebooks folders. When performing inference please take into consideration the the sampling is rather slow.

Other than the implementation of the models, the src folder contains a minimal implementation of an HMC sampler. It can be used with any model and energy function.

### Experiments
This folder contains two types of experiments:
generated_data_recovery - estimating the model's parameters with known ground truth.
cross validation - fitting the models into training data.

Each experiment folder contains a script and configuration file for each model.


### notebooks
This folder contains two notebooks with example of analysis, similar to the one done in the paper.
generated_data_analysis - analysis of parameters recovery.
data_fit_analysis - analysis of the different models fitted to the data. This notebooks contains data generation and may take a while to run fully.

### Results
This folder contains example results of the experiments. The files are used in the analysis in the notebooks.
full_model_generated_data_inference - results of the parameter recovery. Including the generated data and the sampled parameters.
cross_validation - results of the cross validation. One folder for each model. Due to space limitations only one fold (one train-test split) of the cross validation is included.


### DATA
This folder contains processed data. The unprocessed data can be found [here] (https://osf.io/me2sh/).
The folder Saliencies contains a saliency map for each image (128 x 128 pixels).
The folder fixations contains a file for each subject with fixation locations and durations for all images.
The folder processed_data contains the files that are used for the experiments.
