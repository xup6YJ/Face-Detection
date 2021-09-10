
source('~/WireFace/code/4.iterator_2.R')
source('~/WireFace/code/5.architecture.R')
source('~/WireFace/code/6.Support Function.R')

# initiate Parameter for model
new_arg <- mxnet:::mx.model.init.params(symbol = final_yolo_loss, 
                                        input.shape = list(data = c(256, 256, 3, 16), 
                                                           label = c(8, 8, 6, 16)), 
                                        output.shape = NULL, 
                                        initializer = mxnet:::mx.init.Xavier(rnd_type = "uniform", magnitude = 2.24), 
                                        ctx = mx.gpu())

# Bind Pre-trained Parameter into model

Pre_trained_ARG <- Pre_Trained_model$arg.params

ARG_in_net_name <- names(Pre_trained_ARG) %>% .[. %in% names(new_arg$arg.params)]  # remove paramter does not in model

for (i in 1:length(ARG_in_net_name)){
  new_arg$arg.params[names(new_arg$arg.params) == ARG_in_net_name[i]] <- Pre_trained_ARG[names(Pre_trained_ARG) == ARG_in_net_name[i]]
}

ARG.PARAMS <- new_arg$arg.params

## Define fixed layer

Layer_to_fixed <- ARG_in_net_name


# Model Training

my_logger <- mx.metric.logger$new()
my_optimizer <- mx.opt.create(name = "sgd", learning.rate = 5e-3, momentum = 0.9, wd = 1e-4)

my_iter <- my_iterator_func(iter = NULL, 
                            batch_size = 16,
                            img_size = 288, 
                            aug_crop = TRUE, 
                            aug_flip = TRUE,
                            train_val = 'train')

YOLO_model <- mx.model.FeedForward.create(final_yolo_loss, X = my_iter,
                                          ctx = mx.gpu(), 
                                          num.round = 50, 
                                          optimizer = my_optimizer,
                                          arg.params = ARG.PARAMS,  
                                          eval.metric = my.eval.metric.loss,
                                          input.names = 'data', 
                                          output.names = 'label',
                                          batch.end.callback = my.callback_batch(batch.size = 16, frequency = 10),
                                          epoch.end.callback = my.callback_epoch(out_symbol = yolomap, 
                                                                                 logger = my_logger,
                                                                                 prefix = 'WireFace/model', period = 1))


############################################

val_iter <- my_iterator_func(iter = NULL, 
                             batch_size = 32,
                             img_size = 288, 
                             train_val = 'val', 
                             aug_crop = FALSE, 
                             aug_flip = FALSE)

YOLO_model  = my.yolo_trainer(symbol = final_yolo_loss,
                              Iterator_list = my_iter,
                              val_iter = val_iter,
                              ctx = mx.gpu(),
                              num_round = 30,
                              num_iter = 10,
                              start_val = 5,
                              start_unfixed = 5,
                              start.learning_rate = 5e-2,
                              prefix = 'WireFace/model',
                              Fixed_NAMES = NULL,
                              ARG.PARAMS = ARG.PARAMS)
  
  
  
YOLO_model <- mx.model.FeedForward.create(final_yolo_loss, X = my_iter,
                                          ctx = mx.gpu(), num.round = 30, optimizer = my_optimizer,
                                          arg.params = ARG.PARAMS,  eval.metric = my.eval.metric.loss,
                                          input.names = 'data', output.names = 'label',
                                          batch.end.callback = my.callback_batch(batch.size = 16, frequency = 10),
                                          epoch.end.callback = my.callback_epoch(out_symbol = yolomap, 
                                                                                 logger = my_logger,
                                                                                 prefix = 'WireFace/model', period = 1))