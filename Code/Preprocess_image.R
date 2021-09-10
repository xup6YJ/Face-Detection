

load('WireFace/annotation/final_annotation_288.RData')
load('WireFace/annotation/final_annotation_val_288.RData')

train_ids <- unique(annotation[,'img_id'])
val_ids <- unique(val_annotation[,'img_id'])
img_dic = "WireFace/images"

train_img_list = list()
img_idlist = list()
img_list = list()

for (i in 1:length(train_ids)) {
  
  # i = 1
  if(i %% 500 == 0){print(i)}
  
  img_idlist[[i]] = train_ids[i]
  filename <<- paste0(img_dic, '/', train_ids[i])
  read_img <- readJPEG(filename)
  
  if (!dim(read_img)[1] == 288 | !dim(read_img)[2] == 288) {
    
    img_list[[i]] <- resizeImage(image = read_img, 
                                  width = 288, 
                                  height = 288, 
                                  method = 'bilinear')
    } 
  else {
    img_list[[i]] <- read_img
  }
}

train_img_list[[1]] = img_idlist
train_img_list[[2]] = img_list

save(train_img_list, file = 'WireFace/train_img_list_288.RData')

#################################################

img_dic = "WireFace/val_images"
img_idlist = list()
val_img_list = list()
img_list = list()

for (i in 1:length(val_ids)) {
  
  # i = 1
  if(i %% 500 == 0){print(i)}
  
  img_idlist[[i]] = val_ids[i]
  filename <<- paste0(img_dic, '/', val_ids[i])
  read_img <- readJPEG(filename)
  
  if (!dim(read_img)[1] == 256 | !dim(read_img)[2] == 256) {
    
    img_list[[i]] <- resizeImage(image = read_img, 
                                 width = 256, 
                                 height = 256, 
                                 method = 'bilinear')
  } 
  else {
    img_list[[i]] <- read_img
  }
}

val_img_list[[1]] = img_idlist
val_img_list[[2]] = img_list

save(val_img_list, file = 'WireFace/val_img_list_256.RData')
