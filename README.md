# Interleaved_Artifacts_Aware_Attention_Mechanism
This is the official tensorflow implementation of "Interleaved Deep Artifacts-aware Attention Mechanism for Concrete Structural Defect Classification". This repository includes the proposed fine-grained dense module, committee of multi-feature attention module and simultaneous excitation module. The results folder contains the results and the visualizations using attention maps.

## The iDAAM architecture
The novel iDAAM architecture is proposed to classify both multi-target multi-class and singleclass structural defect images. iDAAM architecture consists of interleaved fine-grained dense modules (FGDM) and concurrent dual attention modules (CDAM) to extract salient discriminative features from multiple scales to improve the classification performance. Experimental results and ablation studies show that the newly proposed architecture achieves significantly better classification performance than the current state-of-the-art methodologies on three large datasets.

### Diagram of the iDAAM architecture and the FGDM:
![alt text](https://github.com/NBPuhan/Interleaved_Artifacts_Aware_Attention_Mechanism/blob/main/Figures/iDAAM_full_model.png)

This figure describes (a). the architecture of iDAAM. Here Conv block represents convolutional operation with first number representing the number of filters and the next two numbers give the filter dimension for each channel. For all convolution operations, strides are considered to be one. Dense represents the dense layer, where first number gives the number of nodes and the second number is the dropout value. In the last dense layer, dropout is not used. (b). Proposed concurrent dual attention module composed of committee of multi-feature attention module and simultaneous excitation module. The number denoted in (a) with the concurrent dual attention module represents filter size. (c). Proposed fine-grained dense module. The numbers denoted in (a) with the fine-grained dense module represent number of filters for convolution. The max pooling operation uses pool size of (2,2) and stride 2.

### Diagram of the CDAM (attention module):
![alt text](https://github.com/NBPuhan/Interleaved_Artifacts_Aware_Attention_Mechanism/blob/main/Figures/iDAAM_attention.png)

This figure describes (a). the proposed committee of multi-feature attention module. (b). Concurrent dual attention module. (c). Simultaneous excitation module.
