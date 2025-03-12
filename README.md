[![Python App](https://github.com/laisee/ANN_Option_Pricing/actions/workflows/python-app.yml/badge.svg)](https://github.com/laisee/ANN_Option_Pricing/actions/workflows/python-app.yml)

# ANN OptionPricing
Calculating Option prices with Deep Neural networks(PyTorch)

### Based on following code and papers (plus others)

* https://github.com/abbkey/pricing-options-pytorch
* https://srdas.github.io/DLBook/index.html

* https://arxiv.org/pdf/1901.08943
* https://cs230.stanford.edu/projects_fall_2019/reports/26260984.pdf
* https://github.com/abbkey/pricing-options-pytorch/blob/main/Report.pdf

Sample code showing Options pricing
* https://pt-options-pricer-d878fbfe127c.herokuapp.com

#### 1. Calculating BS price using ANN trained on generated sample data

 - Open terminal app on Mac OSX
 - Download code from this github repo(https://github.com/laisee/ANN_Option_Pricing)
 - Run "pip install -r requirements.txt" to install python tools needed
 - Run "python3 ann_model_01/run_bs_all_sampledata.py"
 - Click on 1st displayed chart to view next one
 - Click on 2nd displayed chart to complete the runnd other metrics for accuracy will be displayed at console
 - Inside the code ("scripts/run_bs_ann_sampledata.py") settings can be updated, such as 
    - SAMPLE_SIZE: number of sample records to generate for training set
    - TRAINING_TESTING_RATIO: split between training & testing data sets e.g. 80% to 20%
    - BATCH_SIZE: size used in training phase
    - LAYER: num ber of layers in model 
    - FEATURES: current training set has five features, update this value if more/less used 
    - NODES: node count for NN model
    - EPOCHS: number of epochs used in training phase

#### 2. Calculating Options pricer using ANN (BS, Heston)
 - Open terminal app on Mac OSX
 - Download code from this github repo(https://github.com/laisee/ANN_Option_Pricing)
 - Run "pip install -r requirements.txt" to install python tools needed
 - Run "python3 ann_model_02/run_model.py "
 - Wait for Epochs(Training) to complete 50 iterations (can be changed in run_model.py script)
 - View chart showing training & testing loss values
 - Inside the code ("ann_model_02/run_model.py") settings can be updated, such as:
    - EPOCHS: number of iteration training phase
