# Face-Detection
Deep Learning approach on Face Detection using YOLOv1

Example Video on Youtube: https://www.youtube.com/watch?v=IBmOdlBjr4c&ab_channel=YJLin

# Data Resource & Citation
Citation: 	
@inproceedings{yang2016wider,
	Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
	Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	Title = {WIDER FACE: A Face Detection Benchmark},
	Year = {2016}}

URL: http://shuoyang1213.me/WIDERFACE/

# Enviornment & Deep Learning Framework
R, MxNet

# Training Instructions

Image Preprocessing & Box Annotation
---
1. Code ["1.Clean Data.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/1.Clean%20Data.R) and is used for original txt cleaning.
- Original Training bbx: wider_face_train_bbx_gt.txt, Validation bbx: wider_face_val_bbx_gt.txt 
- Code for validation set: ["1.Clean Data_val.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/1.Clean%20Data_val.R)

2. Code ["2.Define bbox.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/2.Define%20bbox.R) is used for making standard annotation of bbox.
- Annotate col_left, row_top, bbox_center_col, bbox_center_row of the bbx
- Code for validation set: ["2.Define bbox_val.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/2.Define%20bbox_val.R)

3. Code ["Preprocess_image.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/Preprocess_image.R) is for resizing the images.
- Train data resize in 288x288(For further Random Crop), Validation data resize in 256x256

Encode & Decode Function
---
4. Code ["3.Encode Decode.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/3.Encode%20Decode.R) is function of Ecode, Decode and IOU.

Architecture & Training
---
5. Code ["4.Iterator.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/4.Iterator.R) and is used for cropping and flipping image randomly.

<p align="center">
  <img src="Example Image/example1.jpg">
</p>

6. Code ["5.Architecture.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/5.Architecture.R) and is used for Transfer Learning using MobileNetV2.

7. Code ["7.Train.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/7.Train.R) is used for Training a model.
- Optimizer: SGD, learning rate = 5e-3, momentum  = 0.9, batch size = 16
- Loss Function: Focal Loss(For predicting bbox), Log Cosh (For predicting object)

Model Performance
---
8. Code ["8.MAP.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/8.MAP.R) and is used for calculating MAP.
- We got a result in MAP: 44% under the circumstance of IOU cut : 50%. 

9. Code ["9.Prediction.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/9.Prediction.R) is used for predicting other images which contain faces.

R Shiny UI Server
---
Code ["Server.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Shiny%20UI%20Code/Server.R) and Code ["UI.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Shiny%20UI%20Code/UI.R)is used for building API sever.

<p align="center">
  <img src="Example Image/2.jpg">
</p>
