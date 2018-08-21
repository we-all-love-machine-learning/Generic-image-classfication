## 需要解决的难题

### 迁移学习的局限性
* Network - Method - accuracy of train/val/test - train real-time data augmentation

  * ResNet50 - Transfer Learning with 1 fc layers - 93.04/65.51/65.37 - True
  * InceptionV3 - Fine-tune top 2 inception modules - 96.26/89.57/91.58 - True
  * InceptionResNetV2 - Fine-tune top inception modules and 3 input convolution layers - 96.30/**93.33**/**91.84** - True
  * InceptionResNetV2 - Fine-tune top 2 inception modules - **99.5**/90.91/89.71 - False
  * DenseNet201 - Fine-tune top dense block - 96.42/88.50/88.64 - True
