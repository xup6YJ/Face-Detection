

# Libraries

library(OpenImageR)
library(jpeg)
library(mxnet)
library(imager)
library(magrittr)

# Load data (Training set)
load('WireFace/annotation/final_annotation_288.RData')
load('WireFace/annotation/final_annotation_val_288.RData')
load('WireFace/val_img_list_256.RData')
load('WireFace/train_img_list_288.RData')

head(annotation)

Show_img = function (img, 
                     box_info = NULL,
                     show_prob = FALSE,
                     col_bbox = '#FFFFFF00',
                     col_label = '#FF0000FF',
                     show_grid = FALSE,
                     n.grid = 8,
                     col_grid = '#0000FFFF') {
  
  require(imager)
  
  par(mar = rep(0, 4))
  plot(NA, xlim = c(0.04, 0.96), ylim = c(0.96, 0.04), xaxt = "n", yaxt = "n", bty = "n")
  img <- (img - min(img))/(max(img) - min(img))
  img <- as.raster(img)
  rasterImage(img, 0, 1, 1, 0, interpolate=FALSE)
  
  box_info$col = '#FF0000FF'
  
  box_info[box_info[,'col_left'] < 0, 'col_left'] <- 0
  box_info[box_info[,'col_right'] > 1, 'col_right'] <- 1
  box_info[box_info[,'row_bot'] > 1, 'row_bot'] <- 1
  box_info[box_info[,'row_top'] < 0, 'row_top'] <- 0
  
  for (i in 1:nrow(box_info)) {
    
    # i = 1
    if (is.null(box_info$col[i])) {COL_LABEL <- col_label} else {COL_LABEL <- box_info$col[i]}
    
    if (show_prob) {
      TEXT <- paste0(box_info[i,'obj_name'], ' (', formatC(box_info[i,'prob']*100, 0, format = 'f'), '%)')
    } else {
      TEXT <- box_info[i,'obj_name']
    }
    
    size <- max(box_info[i,'col_right'] - box_info[i,'col_left'], 0.05)
    
    rect(xleft = box_info[i,'col_left'], xright = box_info[i,'col_left'] + 0.04*sqrt(size)*nchar(as.character(TEXT)),
         ybottom = box_info[i,'row_top'] + 0.08*sqrt(size), ytop = box_info[i,'row_top'],
         col = COL_LABEL, border = COL_LABEL, lwd = 0)
    
    text(x = box_info[i,'col_left'] + 0.02*sqrt(size) * nchar(as.character(TEXT)),
         y = box_info[i,'row_top'] + 0.04*sqrt(size),
         labels = TEXT,
         col = 'white', cex = 1.5*sqrt(size), font = 2)
    
    points(box_info$center_x/box_info$org.width, box_info$center_y/box_info$org.height, col="red",pch = 15)
    
    rect(xleft = box_info[i,'col_left'], xright = box_info[i,'col_right'],
         ybottom = box_info[i,'row_bot'], ytop = box_info[i,'row_top'],
         col = col_bbox, border = COL_LABEL, lwd = 5*sqrt(size))
    
  }
  
  if (show_grid) {
    for (i in 1:n.grid) {
      if (i != n.grid) {
        abline(a = i/n.grid, b = 0, col = col_grid, lwd = 12/n.grid)
        abline(v = i/n.grid, col = col_grid, lwd = 12/n.grid)
      }
      for (j in 1:n.grid) {
        text((i-0.5)/n.grid, (j-0.5)/n.grid, paste0('(', j, ', ', i, ')'), col = col_grid, cex = 8/n.grid)
      }
    }
  }
  
}

IoU_function <- function (label, pred) {
  
  overlap_width <- min(label[,2], pred[,2]) - max(label[,1], pred[,1])
  overlap_height <- min(label[,3], pred[,3]) - max(label[,4], pred[,4])
  
  if (overlap_width > 0 & overlap_height > 0) {
    
    pred_size <- (pred[,2]-pred[,1])*(pred[,3]-pred[,4])
    label_size <- (label[,2]-label[,1])*(label[,3]-label[,4])
    overlap_size <- overlap_width * overlap_height
    
    return(overlap_size/(pred_size + label_size - overlap_size))
    
  } else {
    
    return(0)
    
  }
  
}

Encode_fun <- function (box_info, n.grid = 8, eps = 1e-8, obj_name = 'Face') {
  
  
  img_ids <- unique(box_info$img_id)
  num_pred <- 5 + length(obj_name)
  out_array <- array(0, dim = c(n.grid, n.grid, num_pred, length(img_ids)))
  
  for (j in 1:length(img_ids)) {
    
    sub_box_info <- box_info[box_info$img_id == img_ids[j],]
    
    for (i in 1:nrow(sub_box_info)) {
      
      bbox_center_row <- (sub_box_info[i,'row_bot'] + sub_box_info[i,'row_top']) / 2 * n.grid
      bbox_center_col <- (sub_box_info[i,'col_left'] + sub_box_info[i,'col_right']) / 2 * n.grid
      bbox_width <- (sub_box_info[i,'col_right'] - sub_box_info[i,'col_left']) * n.grid
      bbox_height <- (sub_box_info[i,'row_bot'] - sub_box_info[i,'row_top']) * n.grid
      
      center_row <- ceiling(bbox_center_row)
      center_col <- ceiling(bbox_center_col)
      
      row_related_pos <- bbox_center_row %% 1
      row_related_pos[row_related_pos == 0] <- 1
      col_related_pos <- bbox_center_col %% 1
      col_related_pos[col_related_pos == 0] <- 1
      
      out_array[center_row,center_col,1,j] <- 1
      out_array[center_row,center_col,2,j] <- row_related_pos
      out_array[center_row,center_col,3,j] <- col_related_pos
      out_array[center_row,center_col,4,j] <- log(bbox_width + eps)
      out_array[center_row,center_col,5,j] <- log(bbox_height + eps)
      out_array[center_row,center_col,5+which(obj_name %in% sub_box_info$obj_name[i]),j] <- 1 
      
    }
    
  }
  
  return(out_array)
  
}

Decode_fun <- function (encode_array, cut_prob = 0.5, cut_overlap = 0.3,
                        obj_name = 'Face',
                        obj_col = '#FF0000FF',
                        img_id_list = NULL) {
  
  num_img <- dim(encode_array)[4]
  num_feature <- length(obj_name) + 5
  pos_start <- (0:(dim(encode_array)[3]/num_feature-1)*num_feature)
  
  box_info <- NULL
  
  # Decoding
  
  for (j in 1:num_img) {
    
    sub_box_info <- NULL
    
    for (i in 1:length(pos_start)) {
      
      sub_encode_array <- as.array(encode_array)[,,pos_start[i]+1:num_feature,j]
      
      pos_over_cut <- which(sub_encode_array[,,1] >= cut_prob)
      
      if (length(pos_over_cut) >= 1) {
        
        pos_over_cut_row <- pos_over_cut %% dim(sub_encode_array)[1]
        pos_over_cut_row[pos_over_cut_row == 0] <- dim(sub_encode_array)[1]
        pos_over_cut_col <- ceiling(pos_over_cut/dim(sub_encode_array)[1])
        
        for (l in 1:length(pos_over_cut)) {
          
          encode_vec <- sub_encode_array[pos_over_cut_row[l],pos_over_cut_col[l],]
          
          if (encode_vec[2] < 0) {encode_vec[2] <- 0}
          if (encode_vec[2] > 1) {encode_vec[2] <- 1}
          if (encode_vec[3] < 0) {encode_vec[3] <- 0}
          if (encode_vec[3] > 1) {encode_vec[3] <- 1}
          
          center_row <- (encode_vec[2] + (pos_over_cut_row[l] - 1))/dim(sub_encode_array)[1]
          center_col <- (encode_vec[3] + (pos_over_cut_col[l] - 1))/dim(sub_encode_array)[2]
          width <- exp(encode_vec[4])/dim(sub_encode_array)[2]
          height <- exp(encode_vec[5])/dim(sub_encode_array)[1]
          
          if (is.null(img_id_list)) {new_img_id <- j} else {new_img_id <- img_id_list[j]}
          
          new_box_info <- data.frame(obj_name = obj_name[which.max(encode_vec[-c(1:5)])],
                                     col_left = center_col-width/2,
                                     col_right = center_col+width/2,
                                     row_bot = center_row+height/2,
                                     row_top = center_row-height/2,
                                     prob = encode_vec[1],
                                     img_id = new_img_id,
                                     col = obj_col[which.max(encode_vec[-c(1:5)])],
                                     stringsAsFactors = FALSE)
          
          sub_box_info <- rbind(sub_box_info, new_box_info)
          
        }
        
      }
      
    }
    
    if (!is.null(sub_box_info)) {
      
      # Remove overlapping
      
      sub_box_info <- sub_box_info[order(sub_box_info$prob, decreasing = TRUE),]
      
      for (obj in unique(sub_box_info$obj_name)) {
        
        obj_sub_box_info <- sub_box_info[sub_box_info$obj_name == obj,]
        
        if (nrow(obj_sub_box_info) == 1) {
          
          box_info <- rbind(box_info, obj_sub_box_info)
          
        } else {
          
          overlap_seq <- NULL
          
          for (m in 2:nrow(obj_sub_box_info)) {
            
            for (n in 1:(m-1)) {
              
              if (!n %in% overlap_seq) {
                
                overlap_prob <- IoU_function(label = obj_sub_box_info[m,2:5], pred = obj_sub_box_info[n,2:5])
                
                overlap_width <- min(obj_sub_box_info[m,3], obj_sub_box_info[n,3]) - max(obj_sub_box_info[m,2], obj_sub_box_info[n,2])
                overlap_height <- min(obj_sub_box_info[m,4], obj_sub_box_info[n,4]) - max(obj_sub_box_info[m,5], obj_sub_box_info[n,5])
                
                if (overlap_prob >= cut_overlap) {
                  
                  overlap_seq <- c(overlap_seq, m)
                  
                }
                
              }
              
            }
            
          }
          
          if (!is.null(overlap_seq)) {
            
            obj_sub_box_info <- obj_sub_box_info[-overlap_seq,]
            
          }
          
          box_info <- rbind(box_info, obj_sub_box_info)
          
        }
        
      }
      
    }
    
  }
  
  return(box_info)
  
}
#####################################################
# Build an iterator

train_ids <- unique(annotation[,'img_id'])
val_ids <- unique(val_annotation[,'img_id'])
train_box_info = annotation
img_dic = "WireFace/images"
train_img_file = unlist(train_img_list[[1]])
head(train_img_file)
val_img_file = unlist(val_img_list[[1]])
head(train_img_file)

my_iterator_core <- function (batch_size, 
                              img_size = 288, 
                              resize_method = 'nearest',
                              aug_crop = TRUE, 
                              aug_flip = TRUE,
                              train_val = 'train') {
  
  # batch_size = 8
  # img_size = 288
  # resize_method = 'bilinear'
  # aug_crop = TRUE
  # aug_flip = TRUE
  # train_val = 'train'
  
  if(train_val == 'train'){
    train_box_info = train_box_info
    img_dic = "WireFace/images"
    train_ids = train_ids
    image_list = train_img_list
    image_file = train_img_file
  }else{
    train_box_info = val_annotation
    img_dic = "WireFace/val_images"
    train_ids = val_ids
    image_list = val_img_list
    image_file = val_img_file
  }
  
  batch <-  0
  batch_per_epoch <- floor(length(train_ids)/batch_size)
  
  reset <- function() {batch <<- 0}
  
  # batch = 1
  iter.next <- function() {
    
    batch <<- batch + 1
    if (batch > batch_per_epoch) {return(FALSE)} else {return(TRUE)}
    
  }
  
  value <- function() {
    
    idx <- 1:batch_size + (batch - 1) * batch_size
    idx[idx > length(train_ids)] <- sample(1:(idx[1]-1), sum(idx > length(train_ids)))
    idx <- sort(idx)
    
    batch.box_info <<- train_box_info[train_box_info$img_id %in% train_ids[idx],]
    training_id = unique(batch.box_info[,'img_id'])
    
    img_array <- array(0, dim = c(img_size, img_size, 3, batch_size))
    
    for (i in 1:batch_size) {
      
      # i = 1
      pos = which(as.character(image_file) == as.character(training_id[i]))
      read_img <- image_list[[2]][pos][[1]]
      
      # Show_img(read_img)
      
      if (!dim(read_img)[1] == img_size | !dim(read_img)[2] == img_size) {
        
        img_array[,,,i] <- resizeImage(image = read_img, 
                                       width = img_size, 
                                       height = img_size, 
                                       method = resize_method)
        
      } else {
        img_array[,,,i] <- read_img
      }
      
    }
    
    #augmentation
    
    if (aug_flip) {
      
      original_dim <- dim(img_array)
      
      if (sample(0:1, 1) == 1) {
        
        img_array <- img_array[,original_dim[2]:1,,]
        flip_left <- 1 - batch.box_info[,'col_left']
        flip_right <- 1 - batch.box_info[,'col_right']
        batch.box_info[,'col_left'] <- flip_right
        batch.box_info[,'col_right'] <- flip_left
        dim(img_array) <- original_dim
        
      }
      
    }
    
    if (aug_crop) {
      
      revised_dim <- dim(img_array)
      revised_dim[1:2] <- img_size - 32
      
      random.row <- sample(0:32, 1)
      random.col <- sample(0:32, 1)
      
      img_array <- img_array[random.row+1:(img_size-32),random.col+1:(img_size-32),,]
      dim(img_array) <- revised_dim
      
      batch.box_info[,c('row_bot', 'row_top')] <- batch.box_info[,c('row_bot', 'row_top')] * img_size / (img_size - 32) - random.row/img_size
      batch.box_info[,c('col_left', 'col_right')] <- batch.box_info[,c('col_left', 'col_right')] * img_size / (img_size - 32) - random.col/img_size
      

    }
    
    for (j in 9:12) {
      batch.box_info[batch.box_info[,j] <= 0,j] <- 0
      batch.box_info[batch.box_info[,j] >= 1,j] <- 1
    }
    
    label <- Encode_fun(box_info = batch.box_info, n.grid = dim(img_array)[1]/32)
    
    #####test
    # img_seq = 2
    # iter_img <- img_array[,,,img_seq]
    # iter_box_info <- Decode_fun(label)
    # box_info = iter_box_info[iter_box_info$img_id == img_seq,]
    # Show_img(img = iter_img, box_info = box_info, show_grid = FALSE, n.grid = 7)
    
    label <- mx.nd.array(label)
    data <- mx.nd.array(img_array)
    
    return(list(data = data, label = label))
    
  }
  
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch))
  
}

my_iterator_func <- setRefClass("Custom_Iter",
                                fields = c("iter", "batch_size", "img_size", "resize_method", "aug_crop", "aug_flip", "train_val"),
                                contains = "Rcpp_MXArrayDataIter",
                                methods = list(
                                  initialize = function(iter, batch_size = 16, img_size = 256, resize_method = 'nearest',
                                                        aug_crop = TRUE, aug_flip = TRUE, train_val = 'train'){
                                    .self$iter <- my_iterator_core(batch_size = batch_size, img_size = img_size, resize_method = resize_method,
                                                                   aug_crop = aug_crop, aug_flip = aug_flip, train_val = train_val)
                                    .self
                                  },
                                  value = function(){
                                    .self$iter$value()
                                  },
                                  iter.next = function(){
                                    .self$iter$iter.next()
                                  },
                                  reset = function(){
                                    .self$iter$reset()
                                  },
                                  finalize=function(){
                                  }
                                )
)

#########test

my_iter <- my_iterator_func(iter = NULL, 
                            batch_size = 16,
                            img_size = 288, 
                            resize_method = 'bilinear',
                            aug_crop = TRUE, 
                            aug_flip = TRUE,
                            train_val = 'train')

my_iter$reset()

my_iter$iter.next()

test <- my_iter$value()

img_seq = 4

iter_img <- as.array(test$data)[,,,img_seq]

label <- test$label

iter_box_info <- Decode_fun(test$label)
box_info = iter_box_info[iter_box_info$img_id == img_seq,]

Show_img(img = iter_img, box_info = box_info, show_grid = FALSE, n.grid = 7)

######################################
my_iter <- my_iterator_func(iter = NULL, 
                            batch_size = 16,
                            img_size = 256, 
                            resize_method = 'bilinear',
                            aug_crop = FALSE, 
                            aug_flip = FALSE,
                            train_val = 'valid')

my_iter$reset()

my_iter$iter.next()

test <- my_iter$value()

img_seq = 2

iter_img <- as.array(test$data)[,,,img_seq]

label <- test$label

iter_box_info <- Decode_fun(test$label)
box_info = iter_box_info[iter_box_info$img_id == img_seq,]

Show_img(img = iter_img, box_info = box_info, show_grid = FALSE, n.grid = 7)