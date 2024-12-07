# CNN_Project_Histopathologic_Cancer_Detection
Histopathologic Cancer Detection Using Convolutional Neural Network (CNN)

# Histopathalogic Cancer Detection:

The objective of this project is to create an algorithm to identify metastatic cancer in small image patches taken from larger digital pathology scans. The dataset for this project is a slightly modified version of PatchCamelyon (PCAM) benchmark dataset: https://github.com/basveeling/pcam. This project is part of the Kaggle Competition which is detailed in the following link: https://www.kaggle.com/competitions/histopathologic-cancer-detection/overview. For this project specifically, I created a baseline model and two advanced mdoels. 

### Exploratory Data Analysis (EDA): 
The EDA process includes examining the shape of the `labels` dataframe, gathering general information via the `.info()` function, and plotting a histogram to visualize the distribution of each label value. 

## Baseline Model: 

The architecture for the baseline model is detailed below:

1. **Input Layer:** Takes images of shape 32 x 32 pixels with 3 color channels (RGB)

2. **Convolutional Layers:** 
  *  **First Convolutional Layer:** 32 filters with kernel size (3,3). ReLU Activation. 
  *  **Max Pooling Layer:** Reduce spatial dimensions by a factor of 2.
  *  **Second Convolutional Layer:** 64 filters with a kernel size of (3,3) Relu Activation. 

3. **Flatten Layer:** Converts the two dimensional feature maps from the last convolutional layer into a single dimension vector. 

4. **Dense Layers:** 
  * **First Dense Layer:** Fully connected; 128 neurons and ReLU activation.
  * **Output Layer:** Single neuron with a sigmoid activation. 

5. **Optimizer:** Utilizes an Adam Optimizer with a Loss Function of Binary Crossentropy. 

## 1st Advanced Model (Model 1): 

The architecture for the first advanced model (Model 1) is detailed below:

1. **Input Layer:** Takes images of shape 32 x 32 pixels with 3 color channels (RGB)

2. **Convolutional Layers:** 

 *  **First Convolutional Layer:** 32 filters with kernel size (3,3). ReLU Activation. 
 *  **Max Pooling Layer 1:** Reduce spatial dimensions by a factor of 2.
 *  **Second Convolutional Layer:** 64 filters with a kernel size of (3,3) Relu Activation. 
 * **Max Pooling Layer 2:** Reduce spatial dimensions by a factor of 2.
 * **Third Convolutional Layer:** 128 filters with a kernel size of (3,3) Relu Activation. 


3. **Flatten Layer:** Converts the feature maps from the last convolutional layer into a single dimension vector. 

4. **Dense Layers:** 

  * **First Dense Layer:** Fully connected; 128 neurons and ReLU activation.
  * **Output Layer:** Single neuron with a sigmoid activation. 
  * **Dropout:** A dropout value of 0.5 was added to prevent overfitting.

5. **Optimizer:** Utilizes an Adam Optimizer with a Loss Function of Binary Crossentropy.

## 2nd Advanced Model (Model 2): 

The architecture for the second advanced model (Model 2) is detailed below:

1. **Input Layer:** Takes images of shape 32 x 32 pixels with 3 color channels (RGB)

2. **Convolutional Layers:** 

 *  **First Convolutional Layer:** 32 filters with kernel size (3,3). ReLU Activation. 
 *  **Max Pooling Layer 1:** Reduce spatial dimensions by a factor of 2.
 *  **Second Convolutional Layer:** 64 filters with a kernel size of (3,3) Relu Activation. 
 * **Max Pooling Layer 2:** Reduce spatial dimensions by a factor of 2.
 * **Third Convolutional Layer:** 128 filters with a kernel size of (3,3) Relu Activation. 


3. **Flatten Layer:** Converts the feature maps from the last convolutional layer into a single dimension vector. 

4. **Dense Layers:** 

  * **First Dense Layer:** Fully connected; 128 neurons and ReLU activation.
  * **Output Layer:** Single neuron with a sigmoid activation. 
  * **Dropout:** A dropout value of 0.5 was added to prevent overfitting.

5. **Optimizer:** Utilizes an Adam Optimizer with a Loss Function of Binary Crossentropy. The learning rate of the Adam Optimizer was adjusted to 0.0005 based on the Model 1 Loss vs. Epochs graph indicating some fluctuations. 

## Conclusions:

- The model with the 3 convolutional layers and the adjusted learning rate indicated the highest accuracy and lowest validation loss at 0.87 Validation Accuracy and 0.31 Validation Loss. The baseline model performed with an accuracy of 0.84 Validation Accuracy and 0.40 for the Validation Loss. The model with the unadjusted learning rate that included the baseline model with a 3rd convolutional layer and a dropout of 0.5 performed with an accuracy of 0.86 Validation Accuracy and 0.33 Validation Loss. While the Model 2 performance was the best, all three models performed very similarly. 


## Next Steps:

- Additional hyperparameters could be adjusted to provide better accuracy. This could include adjusting the filter size, adding additional convolutional layers, further tuning the learning rate, and much further. Leveraging transfer learning via an advanced CNN architecture, such as ResNet, could also potentially improve the performance of models. 
