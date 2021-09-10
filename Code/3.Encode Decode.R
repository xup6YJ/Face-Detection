
load('WireFace/annotation/final_annotation.RData')

# Custom function

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
  
  # box_info = batch.box_info
  # n.grid = dim(img_array)[1]/32
  # eps = 1e-8
  # obj_name = 'Face'
  
  img_ids <- unique(box_info$img_id)
  num_pred <- 5 + length(obj_name)
  out_array <- array(0, dim = c(n.grid, n.grid, num_pred, length(img_ids)))
  
  for (j in 1:length(img_ids)) {
    
    # j = 1
    sub_box_info <- box_info[box_info$img_id == img_ids[j],]
    
    for (i in 1:nrow(sub_box_info)) {
      
      # i = 1
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
  
  # encode_array = Encode_label
  # cut_prob = 0.5
  # cut_overlap = 0.3
  # obj_name = 'Face'
  # obj_col = '#FF0000FF'
  # img_id_list = NULL
  
  num_img <- dim(encode_array)[4]
  num_feature <- length(obj_name) + 5
  pos_start <- (0:(dim(encode_array)[3]/num_feature-1)*num_feature)
  
  box_info <- NULL
  
  # Decoding
  
  for (j in 1:num_img) {
    
    # j = 1
    sub_box_info <- NULL
    
    for (i in 1:length(pos_start)) {
      
      # i = 1
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


# Test Encode & Decode function

id = levels(annotation$img_id)
length(id)
img_dic = "WireFace/images"

i = 1
sub_BOX_INFOS = annotation[annotation$img_id == id[i],]
resized_img = readJPEG(paste0(img_dic, '/', sub_BOX_INFOS[,'img_id']))

Encode_label <- Encode_fun(box_info = sub_BOX_INFOS)
restore_BOX_INFOS <- Decode_fun(encode_array = Encode_label)

Show_img(img = resized_img, box_info = restore_BOX_INFOS, show_grid = TRUE)


###Resize 288
img_size = 288
x_scale = img_size / annotation$org.height
y_scale = img_size / annotation$org.width

x_pos = annotation$x1 * x_scale
y_pos = annotation$y1 * y_scale
b_width = annotation$w * x_scale
b_height = annotation$h * y_scale

annotation$col_left = x_pos/img_size
annotation$col_right = (x_pos + b_width)/img_size
annotation$row_top = y_pos/img_size
annotation$row_bot = (y_pos + b_height)/img_size


# Test Encode & Decode function

id = levels(annotation$img_id)
length(id)
img_dic = "WireFace/images"

# i = 2
sub_BOX_INFOS = annotation[annotation$img_id == id[i],]
resized_img = readJPEG(paste0(img_dic, '/', sub_BOX_INFOS[,'img_id']))
resized_img = resizeImage(image = resized_img,
                          width = img_size, 
                          height = img_size, 
                          method = resize_method)

Encode_label <- Encode_fun(box_info = sub_BOX_INFOS)
restore_BOX_INFOS <- Decode_fun(encode_array = Encode_label)

Show_img(img = resized_img, box_info = restore_BOX_INFOS, show_grid = FALSE)

save(annotation, file = 'WireFace/annotation/final_annotation_288.RData')