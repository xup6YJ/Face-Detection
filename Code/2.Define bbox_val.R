
library(jpeg)

load('WireFace/annotation/annotation_val.RData')
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
  
  # img = img
  # box_info = info
  
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
##############################################

# "[left, top, width, height, score]"
annotation$x1 = as.numeric(paste(annotation$x1))
annotation$y1 = as.numeric(paste(annotation$y1))
annotation$w = as.numeric(paste(annotation$w))
annotation$h = as.numeric(paste(annotation$h))

annotation$col_left = annotation$x1 / annotation$org.height
annotation$row_top = annotation$y1 / annotation$org.width
annotation$col_right = (annotation$x1 + annotation$w)/annotation$org.height
annotation$row_bot = (annotation$y1 + annotation$h)/annotation$org.width
annotation$obj_name = "Face"
annotation$bbox_center_col = (annotation$col_left + annotation$col_right)/2
annotation$bbox_center_row = (annotation$row_top + annotation$row_bot)/2

#test
id = levels(annotation$img_id)
length(id)
val_img_dic = "WireFace/val_images"

i = 200
info = annotation[annotation$img_id == id[i],]
img = readJPEG(paste0(val_img_dic, '/', info[,'img_id']))
Show_img(img, info)

save(annotation, file = 'WireFace/annotation/final_annotation_val.RData')

###Resize 256
load('WireFace/annotation/final_annotation_val.RData')
img_size = 256
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
val_img_dic = "WireFace/val_images"

i = 2
sub_BOX_INFOS = annotation[annotation$img_id == id[i],]
resized_img = readJPEG(paste0(val_img_dic, '/', sub_BOX_INFOS[,'img_id']))
resized_img = resizeImage(image = resized_img,
                          width = img_size, 
                          height = img_size, 
                          method = 'bilinear')

Encode_label <- Encode_fun(box_info = sub_BOX_INFOS)
restore_BOX_INFOS <- Decode_fun(encode_array = Encode_label)

Show_img(img = resized_img, box_info = restore_BOX_INFOS, show_grid = FALSE)
val_annotation = annotation
save(val_annotation, file = 'WireFace/annotation/final_annotation_val_256.RData')