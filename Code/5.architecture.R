

# Libraries

library(mxnet)
library(magrittr)

## Define the model architecture
## Use pre-trained model and fine tuning

# Load MobileNet v2

Pre_Trained_model <- mx.model.load('model/mobilev2', 0)

# Get the internal output

Mobile_symbol <- Pre_Trained_model$symbol

Mobile_All_layer <- Mobile_symbol$get.internals()

basic_out <- which(Mobile_All_layer$outputs == 'conv6_3_linear_bn_output') %>% Mobile_All_layer$get.output()

# mx.symbol.infer.shape(basic_out, data = c(256, 256, 3, 7))$out.shapes
# conv6_3_linear_bn_output out shape = 8 8 320 n (if input shape = 256 256 3 n)

# Convolution layer for specific mission and training new parameters

# 1. Additional some architecture for better learning

DWCONV_function <- function (indata, num_filters = 256, Inverse_coef = 6, residual = TRUE, name = 'lvl1', stage = 1) {
  
  expend_conv <- mx.symbol.Convolution(data = indata, 
                                       kernel = c(1, 1), 
                                       stride = c(1, 1), 
                                       pad = c(0, 0),
                                       no.bias = TRUE, 
                                       num.filter = num_filters * Inverse_coef,
                                       name = paste0(name, '_', stage, '_expend'))
  
  expend_bn <- mx.symbol.BatchNorm(data = expend_conv, 
                                   fix_gamma = FALSE, 
                                   name = paste0(name, '_', stage, '_expend_bn'))
  
  expend_relu <- mx.symbol.LeakyReLU(data = expend_bn, 
                                     act.type = 'leaky', 
                                     slope = 0.1, 
                                     name = paste0(name, '_', stage, '_expend_relu'))
  
  dwise_conv <- mx.symbol.Convolution(data = expend_relu, 
                                      kernel = c(3, 3), 
                                      stride = c(1, 1), 
                                      pad = c(1, 1),
                                      no.bias = TRUE, 
                                      num.filter = num_filters * Inverse_coef, 
                                      num.group = num_filters * Inverse_coef,
                                      name = paste0(name, '_', stage, '_dwise'))
  
  dwise_bn <- mx.symbol.BatchNorm(data = dwise_conv, fix_gamma = FALSE, name = paste0(name, '_', stage, '_dwise_bn'))
  dwise_relu <- mx.symbol.LeakyReLU(data = dwise_bn, act.type = 'leaky', slope = 0.1, name = paste0(name, '_', stage, '_dwise_relu'))
  
  restore_conv <- mx.symbol.Convolution(data = dwise_relu, 
                                        kernel = c(1, 1), 
                                        stride = c(1, 1),
                                        pad = c(0, 0),
                                        no.bias = TRUE, 
                                        num.filter = num_filters,
                                        name = paste0(name, '_', stage, '_restore'))
  
  restore_bn <- mx.symbol.BatchNorm(data = restore_conv, fix_gamma = FALSE, name = paste0(name, '_', stage, '_restore_bn'))
  
  if (residual) {
    
    block <- mx.symbol.broadcast_plus(lhs = indata, rhs = restore_bn, name = paste0(name, '_', stage, '_block'))
    return(block)
    
  } else {
    
    restore_relu <- mx.symbol.LeakyReLU(data = restore_bn, act.type = 'leaky', slope = 0.1, name = paste0(name, '_', stage, '_restore_relu'))
    return(restore_relu)
    
  }
  
}

CONV_function <- function (indata, num_filters = 256, name = 'lvl1', stage = 1) {
  
  conv <- mx.symbol.Convolution(data = indata, 
                                kernel = c(1, 1), 
                                stride = c(1, 1), 
                                pad = c(0, 0),
                                no.bias = TRUE, 
                                num.filter = num_filters,
                                name = paste0(name, '_', stage, '_conv'))
  
  bn <- mx.symbol.BatchNorm(data = conv, fix_gamma = FALSE, name = paste0(name, '_', stage, '_bn'))
  relu <- mx.symbol.Activation(data = bn, act.type = 'relu', name = paste0(name, '_', stage, '_relu'))
  
  return(relu)
  
}

YOLO_map_function <- function (indata, final_map = 6, num_box = 1, drop = 0.2, name = 'lvl1') {
  
  dp <- mx.symbol.Dropout(data = indata, p = drop, name = paste0(name, '_drop'))
  
  conv <- mx.symbol.Convolution(data = dp, 
                                kernel = c(1, 1),
                                stride = c(1, 1), 
                                pad = c(0, 0),
                                no.bias = FALSE, 
                                num.filter = final_map, 
                                name = paste0(name, '_linearmap'))
  
  inter_split <- mx.symbol.SliceChannel(data = conv, 
                                        num_outputs = final_map,
                                        axis = 1, 
                                        squeeze_axis = FALSE, 
                                        name = paste0(name, "_inter_split"))
  
  new_list <- list()
  
  for (k in 1:final_map) {
    if (!(k %% num_box) %in% c(4:5)) {
      new_list[[k]] <- mx.symbol.Activation(inter_split[[k]], act.type = 'sigmoid', name = paste0(name, "_yolomap_", k))
    } else {
      new_list[[k]] <- inter_split[[k]]
    }
  }
  
  yolomap <- mx.symbol.concat(data = new_list, num.args = final_map, dim = 1, name = paste0(name, "_yolomap"))
  
  return(yolomap)
  
}

yolo_conv_1 <- DWCONV_function(indata = basic_out, num_filters = 320, Inverse_coef = 3, residual = TRUE, name = 'yolo', stage = 1)
yolo_conv_2 <- DWCONV_function(indata = yolo_conv_1, num_filters = 320, Inverse_coef = 3, residual = TRUE, name = 'yolo', stage = 2)
yolo_conv_3 <- CONV_function(indata = yolo_conv_2, num_filters = 320, name = 'yolo', stage = 3)

yolomap <- YOLO_map_function(indata = yolo_conv_3, final_map = 6, drop = 0.2, name = 'final')

# 2. Custom loss function

LOGCOSH_loss_function <- function (indata, inlabel, obj, lambda) {
  
  diff_pred_label <- mx.symbol.broadcast_minus(lhs = indata, rhs = inlabel)
  cosh_diff_pred_label <- mx.symbol.cosh(data = diff_pred_label)
  logcosh_diff_pred_label <- mx.symbol.log(data = cosh_diff_pred_label)
  obj_logcosh_diff_pred_label <- mx.symbol.broadcast_mul(lhs = obj, rhs = logcosh_diff_pred_label)
  LOGCOSH_loss <- mx.symbol.mean(data = obj_logcosh_diff_pred_label, axis = 0:3, keepdims = FALSE)
  
  return(LOGCOSH_loss * lambda)
  
}

Focal_loss_function <- function (indata, inlabel, obj, lambda, gamma = 0, eps = 1e-4) {
  
  log_pred_1 <- mx.symbol.log(data = indata + eps)
  log_pred_2 <- mx.symbol.log(data = 1 - indata + eps)
  multiple_log_pred_label_1 <- mx.symbol.broadcast_mul(lhs = log_pred_1, rhs = inlabel)
  multiple_log_pred_label_2 <- mx.symbol.broadcast_mul(lhs = log_pred_2, rhs = 1 - inlabel)
  obj_weighted_loss <- mx.symbol.broadcast_mul(lhs = obj, rhs = (1 - indata + eps)^gamma * multiple_log_pred_label_1 + (indata + eps)^gamma * multiple_log_pred_label_2)
  average_Focal_loss <- mx.symbol.mean(data = obj_weighted_loss, axis = 0:3, keepdims = FALSE)
  Focal_loss <- 0 - average_Focal_loss * lambda
  
  return(Focal_loss)
  
}

YOLO_loss_function <- function (indata, inlabel, final_map = 6, num_box = 1, lambda = 10, gamma = 2, weight_classification = 0.2, name = 'yolo') {
  
  num_feature <- final_map/num_box
  
  my_loss <- 0
  
  yolomap_split <- mx.symbol.SliceChannel(data = indata,
                                          num_outputs = final_map, 
                                          axis = 1, 
                                          squeeze_axis = FALSE, 
                                          name = paste(name, '_yolomap_split'))
  
  label_split <- mx.symbol.SliceChannel(data = inlabel, 
                                        num_outputs = final_map, 
                                        axis = 1, 
                                        squeeze_axis = FALSE,
                                        name = paste(name, '_label_split'))
  
  for (j in 1:num_box) {
    for (k in 1:num_feature) {
      if (k %in% 1:5) {weight <- 1} else {weight <- weight_classification}
      if (!k %in% c(2:5)) {
        if (k == 1) {
          my_loss <- my_loss + Focal_loss_function(indata = yolomap_split[[(j-1)*num_feature+k]],
                                                   inlabel = label_split[[(j-1)*num_feature+k]],
                                                   obj = label_split[[(j-1)*num_feature+1]],
                                                   lambda = lambda * weight,
                                                   gamma = gamma,
                                                   eps = 1e-4)
          my_loss <- my_loss + Focal_loss_function(indata = yolomap_split[[(j-1)*num_feature+k]],
                                                   inlabel = label_split[[(j-1)*num_feature+k]],
                                                   obj = 1 - label_split[[(j-1)*num_feature+1]],
                                                   lambda = 1,
                                                   gamma = gamma,
                                                   eps = 1e-4)
        } else {
          my_loss <- my_loss + Focal_loss_function(indata = yolomap_split[[(j-1)*num_feature+k]],
                                                   inlabel = label_split[[(j-1)*num_feature+k]],
                                                   obj = label_split[[(j-1)*num_feature+1]],
                                                   lambda = lambda * weight,
                                                   gamma = gamma,
                                                   eps = 1e-4)
        }
      } else {
        my_loss <- my_loss + LOGCOSH_loss_function(indata = yolomap_split[[(j-1)*num_feature+k]],
                                                   inlabel = label_split[[(j-1)*num_feature+k]],
                                                   obj = label_split[[(j-1)*num_feature+1]],
                                                   lambda = lambda * weight)
      }
    }
  }
  
  return(my_loss)
  
}

label <- mx.symbol.Variable(name = "label")

yolo_loss <- YOLO_loss_function(indata = yolomap, 
                                inlabel = label, 
                                final_map = 6, 
                                num_box = 1, 
                                lambda = 10, 
                                gamma = 2, 
                                weight_classification = 0.2, 
                                name = 'yolo')

final_yolo_loss <- mx.symbol.MakeLoss(data = yolo_loss)