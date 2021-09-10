
library(stringr)
library(magrittr)
library(jpeg)
#Read only data

valbbx_path = 'WireFace/wider_face_split/wider_face_val_bbx_gt.txt'
bbxfile = read.table(valbbx_path, sep="\t")

n = nchar(as.character(bbxfile[1,]))
substr(bbxfile[1,], n-3, n)

all_names = NA
for (i in 1:nrow(bbxfile)) {
  
  if(i %% 1000 == 0){print(i)}
  
  n = nchar(as.character(bbxfile[i,]))
  if(n>=4){
    if(substr(bbxfile[i,], n-3, n) == ".jpg"){
      
      name = as.character(bbxfile[i,])
      all_names = rbind(all_names, name)
      
    }
  }
  
}

head(all_names)

all_names = all_names[-1, ]
all_names = as.data.frame(all_names)
save(all_names, file = 'WireFace/annotation/all_names_val.RData')


#Clean Data
annotation = NA
for (j in 1:nrow(bbxfile)) {
  
  if(j %% 1000 == 0){print(j)}
  if(bbxfile[j,] %in% all_names$all_names){
    
    nface = as.numeric(paste(bbxfile[j+1,]))
    
    if(nface>0){
      
      sub_matrix = matrix(0, nrow = nface, ncol = 6)
      
      for (k in 1:nface) {
        
        sub_info = bbxfile[j+1+k,]
        info = str_split(sub_info, " ") %>% unlist()
        
        sub_matrix[k,1] = as.character(bbxfile[j,])
        
        #x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
        #x1, y1, w, h, prob
        sub_matrix[k,2] = as.numeric(info[1])
        sub_matrix[k,3] = as.numeric(info[2])
        sub_matrix[k,4] = as.numeric(info[3])
        sub_matrix[k,5] = as.numeric(info[4])
        sub_matrix[k,6] = 1
      }
    }
    
    sub_matrix = as.data.frame(sub_matrix)
    annotation = rbind(annotation, sub_matrix)
  }
  
}

head(annotation)
annotation = annotation[-1, ]
colnames(annotation) = c('img_id', 'x1', 'y1', 'w', 'h', 'prob')


val_img_dic = "WireFace/val_images"
for (i in 1:nrow(annotation)) {
  
  if(i %% 1000 == 0){print(i)}
  img = readJPEG(paste0(val_img_dic, '/', annotation[i,'img_id']))
  annotation[i,'org.width'] = dim(img)[1]
  annotation[i,'org.height'] = dim(img)[2]
  
}

save(annotation, file = 'WireFace/annotation/annotation_val.RData')