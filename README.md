# Interleaved Artifacts Aware Attention Mechanism
This is the tensorflow implementation of "Interleaved Deep Artifacts-aware Attention Mechanism for Concrete Structural Defect Classification". This repository includes the proposed fine-grained dense module, committee of multi-feature attention module and simultaneous excitation module. The results folder contains the results and the visualizations using attention maps.

# Citation
If you are fully or partially using codes/results from this repository, please cite the following paper: 

G. Bhattacharya, B. Mandal, and N. B. Puhan, "Interleaved Deep Artifacts-aware Attention Mechanism for Concrete Structural Defect Classification". pp. 6957-6969, 2021, doi: 10.1109/TIP.2021.3100556.

The main paper can be accessed from here: https://ieeexplore.ieee.org/document/9505264.

## The iDAAM architecture
The novel iDAAM architecture is proposed to classify both multi-target multi-class and singleclass structural defect images. iDAAM architecture consists of interleaved fine-grained dense modules (FGDM) and concurrent dual attention modules (CDAM) to extract salient discriminative features from multiple scales to improve the classification performance. Experimental results and ablation studies show that the newly proposed architecture achieves significantly better classification performance than the current state-of-the-art methodologies on three large datasets.

### Diagram of the iDAAM architecture and the FGDM:
![alt text](https://github.com/NBPuhan/Interleaved_Artifacts_Aware_Attention_Mechanism/blob/main/Figures/iDAAM_full_model.png)

This figure describes (a). the architecture of iDAAM. Here Conv block represents convolutional operation with first number representing the number of filters and the next two numbers give the filter dimension for each channel. For all convolution operations, strides are considered to be one. Dense represents the dense layer, where first number gives the number of nodes and the second number is the dropout value. In the last dense layer, dropout is not used. (b). Proposed concurrent dual attention module composed of committee of multi-feature attention module and simultaneous excitation module. The number denoted in (a) with the concurrent dual attention module represents filter size. (c). Proposed fine-grained dense module. The numbers denoted in (a) with the fine-grained dense module represent number of filters for convolution. The max pooling operation uses pool size of (2,2) and stride 2.

### Diagram of the CDAM (attention module):
![alt text](https://github.com/NBPuhan/Interleaved_Artifacts_Aware_Attention_Mechanism/blob/main/Figures/iDAAM_attention.png)

This figure describes (a). the proposed committee of multi-feature attention module. (b). Concurrent dual attention module. (c). Simultaneous excitation module.

## Hyper-parameters for training

For our results, 200 epochs are considered with mini-batch size of 16 and learning rate 0.001 and momentum 0.9. The data split protocol same as the existing works were being followed.

## Results

### Classification results on CODEBRIM dataset:
![alt text](https://github.com/NBPuhan/Interleaved_Artifacts_Aware_Attention_Mechanism/blob/main/Results/Quantitative_Results/Table_2.PNG)
### Classification results on SDNET-2018 dataset:
![alt text](https://github.com/NBPuhan/Interleaved_Artifacts_Aware_Attention_Mechanism/blob/main/Results/Quantitative_Results/Table_3.PNG)
### Classification results on Concrete Crack Image dataset:
![alt text](https://github.com/NBPuhan/Interleaved_Artifacts_Aware_Attention_Mechanism/blob/main/Results/Quantitative_Results/Table_4.PNG)

### Attention maps using sample images from three datasets:
![alt text](https://github.com/NBPuhan/Interleaved_Artifacts_Aware_Attention_Mechanism/blob/main/Results/Visualizations/attention_1.PNG)

These figures show the attention maps obtained from the proposed iDAAM network for sample images from CODEBRIM, concrete crack image and SDNET-2018 datasets and are given in top, middle and bottom row, respectively. Original images are followed by their respective attention maps, placed side-by-side. Here, red color denotes highest attention, while blue denotes the lowest attention. Top row from left to right: (a) Exposed bars with mild corrosion, efflorescence and spallation, (b) Corroded bar with spallation, (c) advanced spallation with efflorescence and corrosion, (d) Crack surface with spallation. Middle row from left to right: (e)-(h) Images with crack defects from concrete crack image dataset. Bottom row from left to right: (i) Crack surface in a bridge deck, (j) Cracked wall region, (k) Pavement region with crack, (l) Cracked pavement region.

### Attention maps by gradually adding modules:
![alt text](https://github.com/NBPuhan/Interleaved_Artifacts_Aware_Attention_Mechanism/blob/main/Results/Visualizations/attention_1.PNG)

These figures show the attention maps obtained by gradually adding different sub-networks and using well-known CNN architectures on CODEBRIM dataset. First row, from left to right: (a). Defect image, (b). Only spatial attention, (c). Only channel attention, (d). Only SEM, (e) Only CMFA, (f). CMFA + spatial attention, (g). CMFA + channel attention, (h). iDAAM. Second row, from left to right: (i). Defect image, (j). 1 FGDM + 1 CDAM, (k) 2 FGDM + 2 CDAM, (l). Deep CNN with adaptive thresholding, (m). Inception, (n). ResNet-50, (o). DenseNet-121, (p). iDAAM.
