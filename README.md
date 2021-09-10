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
- Original Training bbx: wider_face_train_bbx_gt.txt, Validation bbx: wider_face_val_bbx_gt.txt 
- Code for validation set: ["1.Clean Data_val.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/1.Clean%20Data_val.R)

2. Code ["2.Define bbox.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/2.Define%20bbox.R) is used for making standard annotation of bbox.
- Annotate col_left, row_top, bbox_center_col, bbox_center_row of the bbx
- Code for validation set: ["2.Define bbox_val.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/2.Define%20bbox_val.R)

3. Code ["Preprocess_image.R"](https://github.com/xup6YJ/Face-Detection/blob/main/Code/Preprocess_image.R) is for resizing the images.
- Train data resize in 288*288, Validation data resize in 256*256

Encode & Decode Function
---

Architecture & Training
---

Model performance
---

R Shiny UI Server
---
