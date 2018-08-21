## 需要解决的难题

### 迁移学习的局限性
* Network - Method - accuracy of train/val/test - train real-time data augmentation

  * ResNet50 - Transfer Learning with 1 fc layer - 93.04/65.51/65.37 - True
  * InceptionV3 - Fine-tune top 2 inception modules - 96.26/89.57/91.58 - True
  * InceptionResNetV2 - Fine-tune top inception modules and 3 input convolution layers - 96.30/**93.33**/**91.84** - True
  * InceptionResNetV2 - Fine-tune top 2 inception modules - **99.5**/90.91/89.71 - False
  * DenseNet201 - Fine-tune top dense block - 96.42/88.50/88.64 - True
  * Xception - Transfer Learning with 1 fc layer - 90.80/83.96/81.28 - True
  * MobileNetV2 - Transefer Learning with 2 fc layers - 89.96/85.83/84.36 - True
  
无论是使用tensorflow-slim-research中提供的最新版本的ckpt模型文件，还是使用keras-applications中的h5模型库， 对于“Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.” Cell, Cell Press, 22 Feb. 2018 所使用的2015.12在imagenet训练且以.pb格式保存的inceptionV3预训练权重来进行DR迁移学习的效果，无法复现。值得一提的是，在相同眼底照下，使用2016.6在imagenet训练并用.ckpt格式保存的InceptionV3预训练权重提取的bottleneck向量，与2015.12的.pb所提取的bottleneck向量并不相同。

由此猜测旧版的预训练权重更适合进行特征提取工作，分类器与特征提取器的耦合性更弱。结合上述表格的测试结果，以及 Gargeya R, Leng T. Automated identiﬁcation of diabetic retinopathy using deep learning. Ophthalmology. 2017;124: 962–969.中使用75137张眼底照训练得到AUC 0.97 TP 0.94 TN 0.98的state-of-the-art的结果与我们使用老版InceptionV3预训练权重所得到acc 94.9的结果，可以认定迁移学习的方法在DR分类中已经走到尽头。

### 新识别方法下的标注工作量

  
 
