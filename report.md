## 需要解决的难题与未来计划

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

由此猜测旧版的预训练权重更适合进行特征提取工作，分类器与特征提取器的耦合性更弱。结合上述表格的测试结果，以及 Gargeya R, Leng T. Automated identiﬁcation of diabetic retinopathy using deep learning. Ophthalmology. 2017;124: 962–969.中使用75137张眼底照训练得到AUC 0.97 TP 0.94 TN 0.98的state-of-the-art的结果与我们使用老版InceptionV3预训练权重所得到acc 94.9的结果相对比，可以认定迁移学习的方法在DR分类中已经走到尽头。

### 新的识别方法的尝试

关于DR分类的性能瓶颈，近两年有学者提出了另外一种解决方式

对于DR病症进行一系列分类，例如
* Classification Method 1
  * normal （无病变）
  * microaneurysms (微小动脉瘤）
  * dot-blot hemorrhages （斑点出血）
  * exudates or cotton wool spots （分泌物或棉絮斑）
  * high-risk lesions such as neovascularization, venous beading, scarring （新生血管，念珠状静脉，伤痕）
  
* Classification Method 2
  * normal
  * microaneurysm
  * hemorrhage
  * excudate
  
眼科医师先按指定分类，对不同级别的DR眼底照进行像素级的分类标注，即将病变区域用方框框起

然后用上述任意效果良好的卷积神经网络，对病变局部图进行分类训练

训练完成，即可利用神经网络标出病变位置，不仅可以识别是否为DR，还能辅助医师判断，提高准确率


**插图**

为了进一步提高分类准确度，Yehui Y, Tao L. Lesion detection and Grading of Diabetic Retinopathy via Two-stages Deep Convolutional Neural Networks. 2017提出了一种方法

基于对病变类型分类的local network得到的病变热点图，再训练一个global network，对DR进行识别或者分期


**插图**


### 新的识别下的标注工作量

Invest Ophthalmol Vis Sci. 2018 Jan 1;59(1):590-596. doi: 10.1167/iovs.17-22721. Retinal Lesion Detection With Deep Learning Using Image Patches中，使用了由2个高级眼科医师标注的243张眼底照，并按病变区域的数量分成了1324张病变区域照。此外，两名医师意见不相符时，结果作废。

Yehui Y, Tao L. Lesion detection and Grading of Diabetic Retinopathy via Two-stages Deep Convolutional Neural Networks. 2017使用了7076张病变区域照，以及重新标注的22795张kaggle病变分期照。值得一提的是，他们寻找了大量的持证眼科医师以及眼科研究生，邀请或付费的方式进行了数据的标注。

**由此可见，标注工作量是十分繁重的**

前者标注数量虽然不大，但标注质量极佳，医师需要投入更大精力

后者标注质量一般，但标注数量很大，投入的人力与金钱资源更大




