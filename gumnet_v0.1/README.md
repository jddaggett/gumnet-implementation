# Gum-Net
This is a modified version of the Geometric unsupervised matching Net-work (Gum-Net) designed to find the geometric correspondence between two images with application to 3D subtomogram alignment and averaging. The third module of the original Gum-Net, the spatial transformer network (STN), is excluded from the implementation here, for experimental purposes. 

The training, unlike the original Gum-Net, is supervised using then ground-truth parameters of the demo dataset for training.

The paper that this code is based on can be found here:

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
