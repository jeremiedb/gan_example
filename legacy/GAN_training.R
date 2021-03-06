#####################################################
### Training module for GAN
#####################################################

devices<- mx.cpu()

input_shape_G<- c(1, 1, 10, batch_size)
input_shape_D<- c(28, 28, 1, batch_size)

mx.metric.binacc <- mx.metric.custom("binacc", function(label, pred) {
  res <- mean(label==round(pred))
  return(res)
})

mx.metric.logloss <- mx.metric.custom("logloss", function(label, pred) {
  res <- mean(label*log(pred)+(1-label)*log(1-pred))
  return(res)
})

##############################################
### Define iterators
iter_G <- G_iterator(batch_size = batch_size)
iter_D <- D_iterator(batch_size = batch_size)

#group<- mx.symbol.Group(linearReg)
exec_G <- mx.simple.bind(symbol = G_sym, data=input_shape_G, ctx = devices, grad.req = "write")
exec_D <- mx.simple.bind(symbol = D_sym, data=input_shape_D, ctx = devices, grad.req = "write")

names(exec_G$arg.arrays)
names(exec_G$grad.arrays)

### initialize parameters - To Do - personalise each layer
param_ini_G<- mxnet:::mx.model.init.params(symbol = G_sym, input.shape = input_shape_G, initializer = mx.init.Xavier(rnd_type = "gaussian", factor_type = "avg", magnitude = 5), ctx = devices)
param_ini_D<- mxnet:::mx.model.init.params(symbol = D_sym, input.shape = input_shape_D, initializer = mx.init.Xavier(rnd_type = "gaussian", factor_type = "avg", magnitude = 5), ctx = devices)

mx.exec.update.arg.arrays(exec_G, param_ini_G$arg.params, match.name=TRUE)
mx.exec.update.aux.arrays(exec_G, param_ini_G$aux.params, match.name=TRUE)

mx.exec.update.arg.arrays(exec_D, param_ini_D$arg.params, match.name=TRUE)
mx.exec.update.aux.arrays(exec_D, param_ini_D$aux.params, match.name=TRUE)

input_names_G <- mxnet:::mx.model.check.arguments(G_sym)
input_names_D <- mxnet:::mx.model.check.arguments(D_sym)


###################################################
#initialize optimizers
optimizer_G<-mx.opt.create(name = "adadelta",
                           rho=0.85, 
                           epsilon = 1e-6, 
                           wd=0, 
                           rescale.grad=1/batch_size, 
                           clip_gradient=1)

updater_G<- mx.opt.get.updater(optimizer = optimizer_G, weights = exec_G$ref.arg.arrays)

optimizer_D<-mx.opt.create(name = "adadelta",
                           rho=0.85, 
                           epsilon = 1e-6, 
                           wd=0, 
                           rescale.grad=1/batch_size, 
                           clip_gradient=1)
updater_D<- mx.opt.get.updater(optimizer = optimizer_D, weights = exec_D$ref.arg.arrays)

####################################
#initialize metric
metric_G<- mx.metric.binacc
metric_G_value<- metric_G$init()
#eval.metric<- metric$init()

metric_D<- mx.metric.binacc
metric_D_value<- metric_D$init()

iteration<- 1
iter_G$reset()
iter_D$reset()

for (iteration in 1:2500) {
  
  iter_G$iter.next()
  iter_D$iter.next()
  
  ### Random input to Generator to produce fake sample
  G_values <- iter_G$value()
  G_data <- G_values[input_names_G]
  mx.exec.update.arg.arrays(exec_G, arg.arrays = G_data, match.name=TRUE)
  mx.exec.forward(exec_G, is.train=T)
  
  ### Feed Discriminator with Concatenated Generator images and real images
  ### Random input to Generator
  D_values_fake <- exec_G$ref.outputs$gact5_output
  D_values_real <- iter_D$value()$data
  
  D_values_fake_split_1<- mx.nd.slice.axis(data = D_values_fake, axis=0, begin = 0, end = batch_size/2)
  D_values_fake_split_2<- mx.nd.slice.axis(data = D_values_fake, axis=0, begin = batch_size/2, end = batch_size)
  
  D_values_real_split_1<- mx.nd.slice.axis(data = D_values_real, axis=0, begin = 0, end = batch_size/2)
  D_values_real_split_2<- mx.nd.slice.axis(data = D_values_real, axis=0, begin = batch_size/2, end = batch_size)
  
  ### Train loop on fake
  mx.exec.update.arg.arrays(exec_D, arg.arrays = list(data=D_values_fake, label=mx.nd.array(rep(0, batch_size))), match.name=TRUE)
  mx.exec.forward(exec_D, is.train=T)
  mx.exec.backward(exec_D)
  update_args_D<- updater_D(weight = exec_D$ref.arg.arrays, grad = exec_D$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec_D, update_args_D, skip.null=TRUE)
  
  metric_D_value <- metric_D$update(label = mx.nd.array(rep(0, batch_size)), exec_D$ref.outputs[["dloss_output"]], metric_D_value)
  
  ### Train loop on real
  mx.exec.update.arg.arrays(exec_D, arg.arrays = list(data=D_values_real, label=mx.nd.array(rep(1, batch_size))), match.name=TRUE)
  mx.exec.forward(exec_D, is.train=T)
  mx.exec.backward(exec_D)
  update_args_D<- updater_D(weight = exec_D$ref.arg.arrays, grad = exec_D$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec_D, update_args_D, skip.null=TRUE)
  
  metric_D_value <- metric_D$update(mx.nd.array(rep(1, batch_size)), exec_D$ref.outputs[["dloss_output"]], metric_D_value)
  
  ### Update Generator weights - use a seperate executor for writing data gradients
  exec_D_back<- mxnet:::mx.symbol.bind(symbol = D_sym, arg.arrays = exec_D$arg.arrays, aux.arrays = exec_D$aux.arrays, grad.reqs = rep("write", length(exec_D$arg.arrays)), ctx = devices)
  mx.exec.update.arg.arrays(exec_D_back, arg.arrays = list(data=D_values_fake, label=mx.nd.array(rep(1, batch_size))), match.name=TRUE)
  mx.exec.forward(exec_D_back, is.train=T)
  mx.exec.backward(exec_D_back)
  D_grads<- exec_D_back$ref.grad.arrays$data
  mx.exec.backward(exec_G, out_grads=D_grads)
  
  update_args_G<- updater_G(weight = exec_G$ref.arg.arrays, grad = exec_G$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec_G, update_args_G, skip.null=TRUE)
  
  ### Update metrics
  #metric_G_value <- metric_G$update(values[[label_name]], exec_G$ref.outputs[[output_name]], metric_G_value)
  
  
  if (iteration %% 25==0){

    D_metric_result <- metric_D$get(metric_D_value)
    #eval_metric_result <- metric$get(eval.metric)
    
    #D_metric_result_log<- paste0("train=", D_metric_result$value)
    #eval_metric_log<- paste0("eval=", eval_metric_result$value)
    
    cat(paste0("[", iteration, "] ", D_metric_result$name, ": ", D_metric_result$value, "\n"))
    #cat(paste0("[", iteration, "] ", train_metric_result$name, ": ", train_metric_log, " ", eval_metric_log, "\n"))
  }
  
  if (iteration ==1 | iteration %% 100==0){
    
    metric_D_value<- metric_D$init()
    
    par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
    for (i in 1:8) {
      img <- as.array(exec_G$ref.outputs$gact5_output)[,,,i]
      plot(as.cimg(img), axes=F)
    }
    
    for (i in 1:8) {
      img <- as.array(iter_D$value()$data)[,,,i]
      plot(as.cimg(img), axes=F)
    }
    
    print(as.numeric(as.array(G_values$digit)))
    
  }
  
}
