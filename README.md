
<p align="center">
  <img width="1000" alt="FM-LCT-title" src="https://github.com/zhenweishi/FM-LCT/assets/17007301/6364d467-8577-4bf7-a6d9-041a3ec11b64">
</p>


## Welcome to Foundation Model for Lung Cancer CT images (FM-LCT).

FM-LCT is a vertical foundation model for quantitative CT analysis in lung cancer. FM-LCT is trained on a diverse dataset covering various lung cancer types and stages, and it undergoes meticulous data preprocessing, ensuring optimal input quality. Harnessing advanced deep learning techniques (contrastive learning algorithms), the model extracts deep learning-based features from CT scans, unveiling crucial insights into morphological features, shape characteristics, and texture features. Positioned for versatile applications in both research and clinical realms, it empowers researchers with nuanced data exploration and provides healthcare professionals with support for informed decision-making.

The FM-LCT is designed and developed by Dr.Zhenwei Shi, Zhihe Zhao, Zhitao Wei, Dr.Chu Han and other AI/CS scientists from [Media Lab](https://github.com/GDPHMediaLab). Also, the work is supported and guided by experienced radiologists Prof. MD Zaiyi Liu and Prof. MD Changhong Liang from the radiolgoy department of Guangdong Provincial People's Hospital.

## Major Features

<p align="center">
<img width="997" alt="FM-LCT-workflow" src="https://github.com/zhenweishi/FM-LCT/assets/17007301/483650da-a22b-42b4-8c68-70c6e3648b9c">
</p>

The FM-LCT model was pre-trained by using constrative learning algorithms (Figure A). The FM-LCT can be implemented to specific use cases by extracting quantitative imaging features or by task-specitic fine-tuning (Figure B). 

## Installation

Before using the FM-LCT foundation model, we suggest users create a vistual environment. Some example codes are as follows:
```
conda create --name fmlct python==3.8
conda activate fmlct
```
Then install related dependencies.

```
pip install -r requirements.txt
```
If the installation runs slowly, you can try the following method

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
## Quick Start (10 mins)

The FM-LCT model can be used in two manners (Figure B). For easy use, we provide an example notebook to describe how it works. The example codes are in Jupyter notebooks.
Note that, users should download the FM-LCT model from the [link](https://drive.google.com/drive/folders/1awQGIi9uXcJuTaOLDkKvG6c-1Vv3ulz_?usp=drive_link) firstly. Also, users need to download dummy data from the [link](https://drive.google.com/drive/folders/1jbP0-lV5tJOBtN2oVBbTfYDs3rcXQodV?usp=drive_link) .
```
[main_directory]/notebooks/[pretrained_checkpoint]
[main_directory]/notebooks/[dataset]
```

## License

This project is freely available to browse, download, and use for scientific and educational purposes as outlined in the [Creative Commons Attribution 3.0 Unported License](https://creativecommons.org/licenses/by/3.0/).

## Disclaimer

FM-LCT is still under development. Although we have tested and evaluated the workflow under many different situations, it may have errors and bugs unfortunately. Please use it cautiously. If you find any, please contact us and we would fix them ASAP.

## Main Developers
 - [Dr. Zhenwei Shi](https://github.com/zhenweishi) <sup/>1, 2
 - MSc. Zhihe Zhao <sup/>2, 3
 - MSc. Zhitao Wei <sup/>2, 4
 - MSc. Xiaodong Zheng <sup/>2, 4
 - [Dr. Chu Han](https://chuhan89.com) <sup/>1, 2
 - MD. Changhong Liang <sup/>1, 2
 - MD. Zaiyi Liu <sup/>1, 2
 

<sup>1</sup> Department of Radiology, Guangdong Provincial People's Hospital (Guangdong Academy of Medical Sciences), Southern Medical University, China <br/>
<sup>2</sup> Guangdong Provincial Key Laboratory of Artificial Intelligence in Medical Image Analysis and Application, China <br/>
<sup>3</sup> School of Medicine, South China University of Technology, China <br/>
<sup>4</sup> Institute of Computing Science and Technology, Guangzhou University, China <br/>

## Contact
We are happy to help you with any questions. Please contact Dr Zhenwei Shi.

We welcome contributions to FM-LCT.
