# DeeplearningClassficationLandsat-8Images
Deep Learning Model For Water/Ice/Land Classification Using Large-Scale Medium Resolution Landsat Satellite Images


Water/Ice/Land region classification is an important remote sensing tasks, which analyze the occurrence of water, ice on the earth surface. Common remote sensing practices such as thresholding, spectral analysis, and statistical approaches generally do not produce globally reliable classification results. Even the robust deep learning models do not perform enough due to the limitation of ground truth available for training and the medium resolution of the Open satellite images. Therefore, in this research, we used a relatively easy method to generate ground truth for randomly selected locations around the globe. Then, we utilized a simplified variant of well-known UNet deep convolutional neural network (CNN) structure with a dilated CNN layers, skip connections and without any max-pooling layers. The proposed model shows better performance in medium resolution satellite images (Landsat-8) compared to state-of-the-art models such as UNet and DeepWaterMap applied on the same task.

**Citation:** P. Vinayaraj, N. Immoglue, R. Nakamura, “Deep learning model for water/ice/land classification using largescale medium resolution satellite images”, IEEE-GRSS International Geoscience and Remote Sensing Symposium (IGARSS), 2019, Yokohama, Japan

**Notes:** 
1. An example dataset is given in the 'data' folder with Landsat meta data file.  
2. list of the Landsat-8 images can be given as'pred_list.txt'. 
3. Six bands are used for classification (Blue, Green, Red, NIR, SWIR1, SWIR2)

