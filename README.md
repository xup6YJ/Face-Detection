# Face-Detection
Deep Learning approach on Face Detection using YOLOv1

# Data Resource & Citation
Citation: 	
@inproceedings{yang2016wider,
	Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
	Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	Title = {WIDER FACE: A Face Detection Benchmark},
	Year = {2016}}

URL:http://shuoyang1213.me/WIDERFACE/

# Enviornment & Deep Learning Framework
R, MxNet

# Training Instructions

Image Preprocessing & Box Annotation
---
1. Code ["1.Clean Data.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/1.Clean%20Data.R) and is used for original txt cleaning.
(Original Training bbx: wider_face_train_bbx_gt.txt, Validation bbx: wider_face_val_bbx_gt.txt["1.Clean Data_val.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/1.Clean%20Data_val.R))

2. Code ["2.Image_preprocessing.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/2.Image_preprocessing.py) is used for customed image function(Crop middle/ Histogram equalization).

3. Code ["3.Train.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/3.Train.py) is for model training using Bootstrapping approach.
4. Code ["4.Evaluation.py"](https://github.com/xup6YJ/Tuberculosis-Classification/blob/main/TB_Example/4.Evaluation.py) is used for concating and evaluating all the bootstrapping results.

Encode & Decode Function
---

Architecture & Training
---

Model performance
---

R Shiny UI Server
---
