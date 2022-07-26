# A Multiscale and Multipath Network with Boundary Enhancement for Building Footprint Extraction from Remotely Sensed Imagery
## The manuscript
 Automatic extraction of building footprints from remotely sensed imagery has become an important means to achieve building distribution due to its high efficiency and low cost, and it can be easily performed by existing fully convolutional network (FCN)-based methods. However, accurately extracting building footprints from remotely sensed imagery remains a challenging task for their existing defects. For example, the cascaded convolutions lead to detailed spatial information losing, producing blurred boundaries and omission of small buildings. Insufficient multiscale features fusion without considering semantic and resolution gaps existing between different level features, which may cause mislabeled pixels. In addition, the insufficient receptive field often produces discontinuous or holey extracted large buildings. To alleviate above problems, we propose a novel multiscale and multipath network with boundary enhancement (MMB-Net) that accurately extracts building footprints from remotely sensed imagery. Specially, a parallel multipath feature extraction module is firstly designed to capture high spatial information preserved multiscale features with less semantic distances. In addition, the receptive field is enlarged and broadened by a multi-scale feature enhancement module. Then, an attention-based multiscale features fusion module is built to appropriately aggregate multiscale features by reducing sematic and resolution gaps and considering different importance of different level features. Lastly, a spatial enhancement module is presented for refining the extracted building boundaries by capturing boundary information from low-level features. MMB-Net was test on two benchmark data sets and compared with some advanced approaches. Experimental results demonstrate that it can achieve promising results.
## Datasets
 * [WHU](http://study.rsgis.whu.edu.cn/pages/download/building_dataset.html "WHU")
 * [SPACENET](https://spacenet.ai/spacenet-buildings-dataset-v2/ "SPACENET")
## Usage
 Clone the repository: git clone https://github.com/lehaifeng/MAPNet.git  
  Hyper-parameters configuration and training are implemented in train.py;  
  The Pytorch implementation of MMB-Net and other related networks are in the model folder;  
  predict.py predict the test dataset, and valid.py evaluate the pixel-level IoU, precision, recall and F1_score metric.  
