# ExplainableAIforMedicalImaging
Explainable AI for Medical Imaging: Study on Alzheimer's Disease using LIME
This project explores the use of explainable artificial intelligence (XAI) techniques for medical imaging analysis, specifically in the context of Alzheimer's disease. The aim of this project is to develop a model that can accurately predict the presence of Alzheimer's disease from brain imaging data, while also providing insights into the reasoning behind its predictions using LIME (Local Interpretable Model-Agnostic Explanations).

Dataset
The dataset used in this project is the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset, which contains MRI and PET imaging data from individuals with and without Alzheimer's disease. The dataset can be downloaded from the ADNI website (http://adni.loni.usc.edu/).

Preprocessing
The imaging data is preprocessed using a combination of tools from the Medical Imaging Interaction Toolkit (MITK) and SimpleITK. This involves image registration, segmentation, and feature extraction.

Model
The model used in this project is a convolutional neural network (CNN) trained on the preprocessed imaging data. The model is designed to predict the presence of Alzheimer's disease from brain imaging data, and also to provide an explanation for its predictions using LIME.

LIME
LIME is a technique for generating local explanations for the predictions of machine learning models. In this project, LIME is used to highlight the regions of the brain that are most important for the prediction, and to provide insights into the reasoning behind the prediction.

Results
The model achieved an accuracy of 85% on a test set of imaging data. The LIME technique used in the model also provided valuable insights into the regions of the brain that are most important for predicting Alzheimer's disease.

Requirements
Python 3.6 or later
TensorFlow 2.0 or later
Keras 2.3 or later
LIME 0.2 or later

Usage
To run the code, first install the required dependencies. Then, download the ADNI dataset and preprocess the imaging data using the provided scripts. Finally, train and evaluate the model using the provided scripts, and use LIME to generate local explanations for the predictions.

Credits
This project was developed by Linda Duamwan as part of Data Science project. 

Reference
chilleos, K.G. et al., 2020. Extracting Explainable Assessments of Alzheimer’s disease via Machine Learning on brain MRI imaging data. 2020 IEEE 20th International Conference on Bioinformatics and Bioengineering (BIBE). 10.1109/bibe50027.2020.00175. 
Akkus, Z. et al., 2017. Deep Learning for Brain MRI Segmentation: State of the Art and Future Directions. Journal of Digital Imaging, 30(4), pp.449–459. Available at: https://link.springer.com/article/10.1007/s10278-017-9983-4. 
Altaf, F. et al., 2019. Going Deep in Medical Image Analysis: Concepts, Methods, Challenges, and Future Directions. IEEE Access, 7, pp.99540–99572. 10.1109/access.2019.2929365. 
B äckström, K. et al., 2018. An Efficient 3D Deep Convolutional Network for Alzheimer’s Disease Diagnosis Using MR Images. IEEE 15th International Symposium on Biomedical Imaging April 4-7. 
Bernal, J., Kushibar, K., Asfaw, D.S., Valverde, S., Oliver, A., Martí, R., Lladó, X., 2018. Deep convolutional neural networks for brain image analysis on magnetic resonance imaging: a review. Artif. Intell. Med., http://dx.doi.org/10.1016/j. Artmed.  
Choi, B.-K. et al., 2020. Convolutional Neural Network-based MR Image Analysis for Alzheimer’s Disease Classification: Current Medical Imaging Formerly Current Medical Imaging Reviews, 16(1), pp.27–35. 10.2174/1573405615666191021123854.  
Ciresan, D., Meier, U., Schmidhuber, J., 2012. Multi-column deep neural networks for image classification. 2012 IEEE Conference on Computer Vision and Pattern Recognition. 10.1109/cvpr.2012.6248110. 
Deepak, S., Ameer, P.M., 2019. Brain tumor classification using deep CNN features via transfer learning. Computers in Biology and Medicine, 111, p.103345. 10.1016/j.compbiomed.2019.103345. 
Deng, L., 2014. Deep Learning: Methods and Applications [online]. Foundations and Trends® in Signal Processing, 7(3-4), pp.197–387. Available at: https://www.nowpublishers.com/article/Details/SIG-039. 
. 
The LIME implementation used in this project is based on the work of Ribeiro et al. (2016) (https://arxiv.org/abs/1602.04938) and https://coderzcolumn.com/tutorials/artificial-intelligence/lime-explain-keras-image-classification-network-predictions.
