library(shiny)
library(DT)
library(jpeg)
library(imager)
library(mxnet)
library(magrittr)
library(OpenImageR)
library(pROC)
library(rlist)


Show_img = function (img, box_info = NULL, col_bbox = '#FFFFFF00', col_label = '#FF0000FF') {
  
  par(mar = rep(0, 4))
  plot(NA, xlim = c(0, 1), ylim = c(1, 0), xaxt = "n", yaxt = "n", bty = "n")
  img = (img - min(img))/(max(img) - min(img))
  img = as.raster(img)
  rasterImage(img, 0, 1, 1, 0, interpolate=FALSE)
  
  
  if (!is.null(box_info)) {
    if (nrow(box_info) > 0) {
      for (i in 1:nrow(box_info)) {
        size = max(box_info[i,3] - box_info[i,2], 0.2)
        rect(xleft = box_info[i,2], xright = box_info[i,2] + 0.06*sqrt(size)*nchar(box_info[i,1]),
             ybottom = box_info[i,5] + 0.08*sqrt(size), ytop = box_info[i,5],
             col = col_label, border = col_label, lwd = 0)
        text(x = box_info[i,2] + 0.03*sqrt(size) * nchar(box_info[i,1]),
             y = box_info[i,5] + 0.04*sqrt(size),
             labels = box_info[i,1],
             col = 'white', cex = 1.5*sqrt(size), font = 2)
        rect(xleft = box_info[i,2], xright = box_info[i,3],
             ybottom = box_info[i,4], ytop = box_info[i,5],
             col = col_bbox, border = col_label, lwd = 5*sqrt(size))
      }
    }
  }
  
}

Show_orgimg = function (img) {
  
  require(imager)
  
  par(mar = rep(0, 4))
  plot(NA, xlim = c(0.04, 0.96), ylim = c(0.96, 0.04), xaxt = "n", yaxt = "n", bty = "n")
  img <- (img - min(img))/(max(img) - min(img))
  img <- as.raster(img)
  rasterImage(img, 0, 1, 1, 0, interpolate=FALSE)
  
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

Decode_fun <- function (encode_array, 
                        cut_prob = 0.5,
                        cut_overlap = 0.3,
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

my_predict <- function (model = YOLO_model, img, ctx = mx.gpu()) {
  
  require(magrittr)
  pred_out <- mxnet:::predict.MXFeedForwardModel(model = YOLO_model, X = img)
  pred_box_info <- Decode_fun(pred_out, cut_prob = 0.5, cut_overlap = 0.3)
  
  return(pred_box_info)
  
}

app_crop_predict <- function(indata_1 , 
                             indata_2 , 
                             img_size = 72, 
                             crop_img_size = 64,
                             dis_model = res_model, 
                             dis_cutpoint = 0.5878633, 
                             ctx = mx.gpu(), 
                             batch_size = 1) {
  
  #2. build a dis exec
  dis_layers <- dis_model$symbol$get.internals()
  dis_model_symbol <- which(dis_layers$outputs == 'high_feature_output') %>% dis_layers$get.output()
  arg_lst <- list(symbol = dis_model_symbol, ctx = ctx, grad.req = 'null', data = c(crop_img_size, crop_img_size, 3, batch_size))
  dis_pred_exc <- do.call(mx.simple.bind, arg_lst)
  
  mx.exec.update.arg.arrays(dis_pred_exc, dis_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(dis_pred_exc, dis_model$aux.params, match.name = TRUE)
  
  #3. predict
  
  X1 <- array(data = indata_1, dim = c(img_size, img_size, 3, batch_size))
  X2 <- array(data = indata_2, dim = c(img_size, img_size, 3, batch_size))
  
  batch_dis <- list()
  sub_X1_list <- list()
  sub_X2_list <- list()
  
  sub_X1_list[[1]] = X1[1:64,1:64,,1] #lefttop
  sub_X1_list[[2]] = X1[9:72,1:64,,1] #righttop
  sub_X1_list[[3]] = X1[1:64,9:72,,1] #leftbottom
  sub_X1_list[[4]] = X1[9:72,9:72,,1] #rightbottom
  sub_X1_list[[5]] = X1[5:68,5:68,,1] #center
  
  sub_X2_list[[1]] = X2[1:64,1:64,,1] #lefttop
  sub_X2_list[[2]] = X2[9:72,1:64,,1] #righttop
  sub_X2_list[[3]] = X2[1:64,9:72,,1] #leftbottom
  sub_X2_list[[4]] = X2[9:72,9:72,,1] #rightbottom
  sub_X2_list[[5]] = X2[5:68,5:68,,1] #center
  
  for (m in 1:5) {
    
    batch_SEQ_ARRAY_1 <- array(sub_X1_list[[m]], dim = c(crop_img_size, crop_img_size, 3, batch_size))
    mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(data = mx.nd.array(batch_SEQ_ARRAY_1)), match.name = TRUE)
    mx.exec.forward(dis_pred_exc, is.train = FALSE)
    X1_batch_predict_out <- as.array(dis_pred_exc$ref.outputs[[1]])
    
    batch_SEQ_ARRAY_2 <- array(sub_X2_list[[m]], dim = c(crop_img_size, crop_img_size, 3, batch_size))
    mx.exec.update.arg.arrays(dis_pred_exc, arg.arrays = list(data = mx.nd.array(batch_SEQ_ARRAY_2)), match.name = TRUE)
    mx.exec.forward(dis_pred_exc, is.train = FALSE)
    X2_batch_predict_out <- as.array(dis_pred_exc$ref.outputs[[1]])
    
    batch_dis[[m]] <- (X1_batch_predict_out - X2_batch_predict_out)^2 %>% colSums(., dims = 3) %>% sqrt(.)
    
  }
  
  dis_mean <<- (batch_dis[[1]] + batch_dis[[2]] + batch_dis[[3]] + batch_dis[[4]] + batch_dis[[5]]) / 5
  return(dis_mean)
  
}

list_append = function(o_list, add_list){
  n_list = list()
  for (i in 1:length(add_list)) {
    names = c('name', 'image')
    n_list[[names[i]]] = add_list[[i]]
  }
  
  return(n_list)
}

YOLO_model <- mx.model.load("./model", iteration = 32)
res_model = mx.model.load(paste0("train v", 60), 60)
load('image_list.RData')

############################################
shinyServer(function(input, output) {
  
  #Page1
  IMAGE = reactive({
    if (input$mark_pic == "Upload") {
      if(is.null(input$files)){return()}else{
        img = readJPEG(input$files$datapath)
        return(img)} 
    }else {
      path = paste0('image/',input$mark_pic, '.jpg')
      m_img = readJPEG(path)
      return(m_img)
    }
  })
  
  MY_TABLE = reactiveValues(table = NULL)

  observeEvent(input$plot_dblclick, {
    
    brush = input$plot_brush
    if (!is.null(brush) & !is.null(input$files$name)) {
      new_table = data.frame(obj_name = input$name,
                             col_left = brush$xmin,
                             col_right = brush$xmax,
                             row_bot = brush$ymax,
                             row_top = brush$ymin,
                             prob = 1,
                             img_id = input$files$name,
                             stringsAsFactors = FALSE)
      MY_TABLE$table = rbind(MY_TABLE$table, new_table)
      
    }else if(!is.null(brush) & input$mark_pic == "Example"){
      new_table = data.frame(obj_name = input$name,
                             col_left = brush$xmin,
                             col_right = brush$xmax,
                             row_bot = brush$ymax,
                             row_top = brush$ymin,
                             prob = 1,
                             img_id = input$mark_pic,
                             stringsAsFactors = FALSE)
      MY_TABLE$table = rbind(MY_TABLE$table, new_table)
  
    }
  })
  
  box = reactive({
    if (input$mark_pic == "Upload") {
      box_info = MY_TABLE$table
      box_info = box_info[box_info[,"img_id"] == input$files$name,]
    }else if(input$mark_pic == "Example"){
      box_info = MY_TABLE$table
      box_info = box_info[box_info[,"img_id"] == input$mark_pic,]
    }else{
      box_info = NULL
    }
    return(box_info)
  })
    
    
  face_img = reactive({
    
    box_info = MY_TABLE$table
    img = IMAGE()
    
    # dim height width
    box_info[,c(2:3)] = as.integer(box_info[,c(2:3)]*dim(img)[2])  #width
    box_info[,c(4:5)] = as.integer(box_info[,c(4:5)]*dim(img)[1])  #height
    
    #row_top, row_bot, col_left, col_right
    out_img = img[box_info[,5]:box_info[,4], box_info[,2]:box_info[,3],]
    out_img <- resizeImage(image = out_img, width = 72, height = 72, method = 'bilinear')

    return(out_img)
  })
  
  output$plot = renderPlot({
    
    img = IMAGE()
    box_info = box()
    if (is.null(img)) {return()} else {
      Show_img(img = img, box_info = box_info)
    }
  })
  
  observeEvent(input$delete, {
    selection = as.numeric(input$table_rows_selected)
    if (length(selection)!=0) {
      MY_TABLE$table = MY_TABLE$table[-selection,]
    }
  })
  
  my_list = reactiveValues(name = NULL, image = NULL)
  
  observeEvent(input$Show, {
    
    output$box_plot = renderPlot({

      img = face_img()
      if(is.null(img)){
        return()
      }else{
        Show_orgimg(img = img)
      }
    })
    
      box_info = MY_TABLE$table
      img = face_img()
      my_list$name = box_info[,1]
      my_list$image = as.array(img)
      reactiveValuesToList(my_list)
    
  })
  
  output$table = DT::renderDataTable({
    dat = MY_TABLE$table
    if (is.null(dat)) {return()} else {
      dat[,2] = round(dat[,2], 3)
      dat[,3] = round(dat[,3], 3)
      dat[,4] = round(dat[,4], 3)
      dat[,5] = round(dat[,5], 3)
      Result = DT::datatable(dat)
      return(Result)
    }
  })
  
  ###################
  #save data
  savecsvData <- function(data) {
    # Create a unique file name
    fileName <- 'markface.csv'
    # Write the file to the local system
    write.csv(data, fileName, row.names = FALSE)
  }
  
  observeEvent(input$Save, {
    
    dat = box()
    add_list = isolate(reactiveValuesToList(my_list))
    
    if(file.exists('markface.csv')){
      
      o_dat = read.csv('markface.csv')
      dat = rbind(o_dat, dat)
    }
    savecsvData(data = dat)
    
    if(file.exists('image_list.RData')){

      load('image_list.RData')
      image_list = list.append(image_list, add_list)
    }
    listname = 'image_list.RData'
    save(image_list, file = listname)
  })
  
  ###################
  
  output$download = downloadHandler(
    filename = function() {'label.csv'},
    content = function(con) {
      dat = MY_TABLE$table
      if (is.null(dat)) {return()} else {
        write.csv(dat, con, row.names = FALSE)
      }
    }
  )
  
  ##############################################
  
  #Page2
  input_img = reactive({
    if (input$Picture == "Upload") {
      if(is.null(input$detection_files)){return()}else{
        img = readJPEG(input$detection_files$datapath)
        return(img)} 
    }else {
      path = paste0('image/',input$Picture, '.jpg')
      exp_img = readJPEG(path)
      return(exp_img)
    }
  })
  
  resize_image<-reactive({
    img = input_img()
    if (is.null(img)) {return()} else {
      img_array <- resizeImage(image = img, width = 256, height = 256, method = 'bilinear')
      dim(img_array) <- c(256, 256, 3, 1)
      return(img_array)
    }
  })
  
  my_prediction<-reactive({
    img_array = resize_image()
    if (is.null(img)){
      return()
    }else{
      pred <- my_predict(model = YOLO_model, img = img_array, ctx = mx.gpu())
      return(pred)
    }
  })
  
  #Step 2 for classification
  dis_prediction<-reactive({

    pred = my_prediction()
    img_array = resize_image()

    pred_box_int = as.integer(pred[,c(2:5)]*256)
    
    #row_top, row_bot, col_left, col_right
    out_img = img_array[pred_box_int[4]:pred_box_int[3], pred_box_int[1]:pred_box_int[2],,]
    out_img <- resizeImage(image = out_img, width = 72, height = 72, method = 'bilinear')
    # dim(out_img) = c(dim(out_img)[1], dim(out_img)[2], dim(out_img)[3])

    dis <- array(data = NA, dim = c(length(image_list), 2))
    colnames(dis) <- c("indata_2_name", "dis")

    if (is.null(pred) & is.null(out_img)){
      return()
    }else{
      
      # Calculate high feature distance for each image in database
      for (i in 1:length(image_list)) {
        
        dis[i, "indata_2_name"] <- image_list[[i]]$name
        dis[i, "dis"] <- app_crop_predict(indata_1 = out_img, 
                                          indata_2 = image_list[[i]]$image, 
                                          img_size = 72, 
                                          crop_img_size = 64,
                                          dis_cutpoint = 0.5878633, 
                                          ctx = mx.gpu(),
                                          batch_size = 1)
      }
    }

    return(dis)

  })
  
  observeEvent(input$Show2, {
    
    output$bbx_result<-renderPrint({
      pred = my_prediction()
      return(pred)
    })
    
    output$prediction_image<-renderPlot({
      
      pred = my_prediction()
      img_array = resize_image()
      dim(img_array) = c(dim(img_array)[1], dim(img_array)[2], dim(img_array)[3])
      
      if (is.null(img_array) & is.null(pred)){
        return()
      }else{
        Show_img(img = img_array, box_info = pred)
      }
    })
    
    output$Summary = DT::renderDataTable({
      dat = my_prediction()
      if (is.null(dat)) {return()} else {
        dat[,2] = round(dat[,2], 3)
        dat[,3] = round(dat[,3], 3)
        dat[,4] = round(dat[,4], 3)
        dat[,5] = round(dat[,5], 3)
        dat[,6] = round(dat[,6], 3)
        Result = DT::datatable(dat[,1:6])
        return(Result)
      }
    })
    
    output$class_person = renderText({
      
      distance = dis_prediction()
      
      #find person
      dis_cutpoint = 0.5878633
      
      if(min(distance[,"dis"]) > dis_cutpoint) {
        return("Identity Unknown, will be added to the database.")
      } else {
        similar_row <- which(distance[,"dis"] == min(distance[,"dis"]))
        return(paste0("Identity:", distance[similar_row, "indata_2_name"]))
      }
      
    })

    
  })
  
  ##############################################
  
  #Page3
  MY_TABLE2 = reactiveValues(table = NULL)

  output$face_dataframe = DT::renderDataTable({
    if(file.exists('markface.csv')){
      o_dat = read.csv('markface.csv')
    }
    Result = DT::datatable(o_dat)
    return(Result)
  })
  
  # observeEvent(input$delete_base, {
  #   
  #   o_dat = read.csv('markface.csv')
  #   MY_TABLE2 = rbind(MY_TABLE2, o_dat)
  #   selection = as.numeric(input$table_rows_selected)
  #   if (length(selection)!=0) {
  #     MY_TABLE2$table = MY_TABLE2$table[-selection,]
  #     savecsvData(data = MY_TABLE2)
  #   }
  # })
  ##############################################
  
  #Page4
  output$linchin_image<-renderPlot({
    img = readJPEG('image/linchin.jpg')
    dim(img) = c(dim(img)[1], dim(img)[2], dim(img)[3])
    
    if (is.null(img)){
      return()
    }else{
      Show_orgimg(img = img)
    }
  })
  
  output$linchin_text<-renderText({
    text = 'Instructor\n
    林嶔 博士(Chin Lin)\n
    國防醫學院 生命科學研究所 助理教授\n
    三軍總醫院 人工智慧暨物聯網中心 技術長\n
    國防醫學院 醫學科技教育中心 副主任'
  })
  
  #YC
  output$yc_image<-renderPlot({
    img = readJPEG('image/yc.jpg')
    dim(img) = c(dim(img)[1], dim(img)[2], dim(img)[3])
    
    if (is.null(img)){
      return()
    }else{
      Show_orgimg(img = img)
    }
  })
  
  output$yc_text<-renderText({
    text = 'Postgraduate\n
    陳映竹 研究生(Ying Chu Chen)\n
    國防醫學院 公共衛生研究所\n
    高雄醫學大學 公共衛生學系\n
    國立台灣師範大學附屬高級中學\n
    負責任務：人臉身份識別'
  })
  
  #YJ
  output$yj_image<-renderPlot({
    img = readJPEG('image/yj.jpg')
    dim(img) = c(dim(img)[1], dim(img)[2], dim(img)[3])
    
    if (is.null(img)){
      return()
    }else{
      Show_orgimg(img = img)
    }
  })
  
  output$yj_text<-renderText({
    text = 'Postgraduate\n
    林穎志 研究生(YingJhi Lin)\n
    國防醫學院 公共衛生研究所-健康科技組\n
    高雄醫學大學 醫管暨醫資學系 \n
    台北市立成功高級中學\n
    負責任務: 人臉物件識別、UI介面'
  })
  
  
})
