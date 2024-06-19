# Gum-Net
Geometric unsupervised matching Net-work (Gum-Net) finds the geometric correspondence between two images with application to 3D subtomogram alignment and averaging. We introduce an end-to-end trainable architecture with three novel modules specifically designed for preserving feature spatial information and propagating feature matching information. 

<p align="center">
<img src="https://user-images.githubusercontent.com/31047726/84725693-2ec78800-af59-11ea-94a3-fdd6b5242645.png" width="800">
</p>

The training is performed in a fully unsupervised fashion to optimize a matching metric. No ground truth transformation information nor category-level or instance-level matching supervision information is needed. As the first 3D unsupervised geometric matching method for images of strong transformation variation and high noise level, Gum-Net significantly improved the accuracy and efficiency of subtomogram alignment. 

<p align="center">
<img src="https://user-images.githubusercontent.com/31047726/84724490-536e3080-af56-11ea-93b8-b31bd4f18cd6.gif" width="400">
</p>

Please refer to our paper for more details:

Zeng, Xiangrui, and Min Xu. "Gum-Net: Unsupervised Geometric Matching for Fast and Accurate 3D Subtomogram Image Alignment and Averaging." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4073-4084. 2020. [[CVPR 2020 open access](http://openaccess.thecvf.com/content_CVPR_2020/html/Zeng_Gum-Net_Unsupervised_Geometric_Matching_for_Fast_and_Accurate_3D_Subtomogram_CVPR_2020_paper.html)]


## Package versions
* torch==2.3.0

## Note
This version of the README is modified from the original in the aitom repo

## Installation 
No installations necessary for this version

## Demo

### Dataset

The [demo dataset](https://cmu.box.com/s/la07ke48s6vkv8y4ntv7yn1hlgwo9ybn) consists of 100 subtomogram pairs (20 of each structure) simulated at SNR 0.1. Transformation ground truth is provided for evaluation. 

Masks of observed region and missing region in Fourier space are provided for imputation in the spatial transformation step. Tilt angle range masks can be generated using functions in aitom.image.vol.wedge.util.

### Trained model

The model is trained by running train_demo.py

### Training code

The training code finetunes the trained model (from SNR 100 dataset) on the demo dataset (SNR 0.1) for 20 iterations. 

```
python train_demo.py
```

Output:

```
Before Finetuning: 
Rotation error:  1.6623960068954804 +/- 0.7343824644662732 Translation error:  8.788284627991649 +/- 3.3609601749211024 ----------

Training Iteration 0
......
......
......
Training Iteration 19
......

After Finetuing:
Rotation error:  1.613769178833486 +/- 0.7337544707569074 Translation error:  0.8483087480267897 +/- 0.6085821785274531 ----------
```


### BibTeX

If you use or modify the code from this project in your project, please cite:
```bibtex
@inproceedings{zeng2020gum,
  title={Gum-Net: Unsupervised Geometric Matching for Fast and Accurate 3D Subtomogram Image Alignment and Averaging},
  author={Zeng, Xiangrui and Xu, Min},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4073--4084},
  year={2020}
}
```
Thank you!
