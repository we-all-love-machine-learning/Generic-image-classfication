# Generic-image-classfication
Generic image classification for custom dataset with various models, implemented by Keras

Multi-gpu is supported.

## Requirments
* Python 3.5 or 3.6
* Tensorflow >= 1.8.0
* Keras >= 2.2.0

## Preprocess
**No need to change the resolution of the images**

**Automatic data augmenatation is implemented in this library**

Split custom dataset into following structure
* train
  * class1
  * class2
* val
  * class1
  * class2
* test
  * class1
  * class2

## Available Models
### InceptionV3, InceptionResNetV2
* **retrain** mode: Randomly initialize all layers and retrain the whole model.

* **finetune** mode: Train last 2 inception blocks.

* **transfer** mode: Train only fully connected layer(s).

### DenseNet201
* **retrain** mode: Randomly initialize all layers and retrain the whole model.

* **finetune** mode: Train last Dense blocks.

* **transfer** mode: Train only fully connected layer(s).

### ResNet50, Xception, MobileNetV2
* **retrain** mode: Randomly initialize all layers and retrain the whole model.

* finetune mode: **TODO**

* **transfer** mode: Train only fully connected layer(s).

### NASNet-Large
**There is an official bug about layer numbers.**

## Hyper-parameters
### Default
* batch size: **32**
* epoch: **50**
* fc layer: **1**
* data augmentation: **False**
* optimizer: **'Adadelta'**
* gpu: **1**
