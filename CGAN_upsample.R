#### GAN on MNIST

rm(list = ls())
require("imager")
require("data.table")
require("mxnet")

######################################################
### Data import and preperation
### First download MNIST train data at Kaggle: 
###   https://www.kaggle.com/c/digit-recognizer/data
######################################################
train <- fread('data/train.csv')
train <- data.matrix(train)

train_data <- train[,-1]
train_data <- t(train_data/255)
train_label <- as.integer(train[,1])

dim(train_data) <- c(28, 28, 1, ncol(train_data))

##################################################
#### Model parameters
##################################################
noise_size <- 128
gen_features <- 64
dis_features <- 32
embed_size <- 2
image_depth <- 1
fix_gamma <- T
no_bias <- T
eps <- 1e-5
batch_size <- 64

##################################################
#### Generator Symbol
##################################################
label <- mx.symbol.Variable('label')
label <- mx.symbol.one_hot(indices = label, depth = 10, name = "label_one_hot")
label <- mx.symbol.reshape(data = label, shape = c(1, 1, 10, batch_size), name = "label_reshape")

g_noise <- mx.symbol.normal(loc=0, scale=1, shape=c(1, 1, noise_size, batch_size), name="gen_rand")
g0 <- mx.symbol.concat(data = c(label, g_noise), num.args = 2, name="gen_concat")

# Optional initial projection [1,1] -> [1,1]
g0 <- mx.symbol.Convolution(data=g0, kernel=c(1,1), stride=c(1,1), pad=c(0,0), num_filter=gen_features*4, no_bias=no_bias)
g0 <- mx.symbol.BatchNorm(g0, fix_gamma=fix_gamma, eps=eps)
g0 <- mx.symbol.relu(g0)

# [1,1] -> [2,2]
g0 <- mx.symbol.UpSampling(g0, scale = 2, sample_type = "nearest", num_args = 1, name='g0_up')
g0 <- mx.symbol.Convolution(data=g0, kernel=c(3,3), stride=c(1,1), pad=c(1,1), num_filter=gen_features*4, no_bias=no_bias, name='g0_conv')
g0 <- mx.symbol.BatchNorm(g0, fix_gamma=fix_gamma, eps=eps, name='g0_bn')
g0 <- mx.symbol.relu(g0, name='g0_act')

# [2,2] -> [4,4]
g1 <- mx.symbol.UpSampling(g0, scale = 2, sample_type = "nearest", num_args = 1, name='g1_up')
g1 <- mx.symbol.Convolution(data=g1, kernel=c(3,3), stride=c(1,1), pad=c(1,1), num_filter=gen_features*2, no_bias=no_bias, name='g1_conv')
g1 <- mx.symbol.BatchNorm(g1, fix_gamma=fix_gamma, eps=eps, name='g1_bn')
g1 <- mx.symbol.relu(g1, name='g1_act')

# [4,4] -> [8,8]
g2 <- mx.symbol.UpSampling(g1, scale = 2, sample_type = "nearest", num_args = 1, name='g2_up')
g2 <- mx.symbol.Convolution(data=g2, kernel=c(3,3), stride=c(1,1), pad=c(1,1), num_filter=gen_features*2, no_bias=no_bias, name='g2_conv')
g2 <- mx.symbol.BatchNorm(g2, fix_gamma=fix_gamma, eps=eps, name='g2_bn')
g2 <- mx.symbol.relu(g2, name='g2_act')

# [8,8] -> [16,16] -> [14,14]
g3 <- mx.symbol.UpSampling(g2, scale = 2, sample_type = "nearest", num_args = 1, name='g3_up')
g3 <- mx.symbol.Convolution(data=g3, kernel=c(3,3), stride=c(1,1), pad=c(0,0), num_filter=gen_features, no_bias=no_bias, name='g3_conv')
g3 <- mx.symbol.BatchNorm(g3, fix_gamma=fix_gamma, eps=eps, name='g3_bn')
g3 <- mx.symbol.relu(g3, name='g3_act')

# [14,14] -> [28,28]
g4 <- mx.symbol.UpSampling(g3, scale = 2, sample_type = "nearest", num_args = 1, name='g4_up')
g4 <- mx.symbol.Convolution(data=g4, kernel=c(3,3), stride=c(1,1), pad=c(1,1), num_filter=1, no_bias=no_bias, name='g4_conv')
G_sym <- mx.symbol.sigmoid(g4, name='G_sym')


##################################################
#### Discriminator Symbol
##################################################
data <- mx.symbol.Variable('data')
digit <- mx.symbol.Variable('digit')
real_flag <- mx.symbol.Variable('real_flag')

digit <- mx.symbol.one_hot(indices = digit, depth = 10, name = "digit_one_hot")
digit <- mx.symbol.reshape(data = digit, shape = c(1, 1, 10, batch_size), name="digit_reshape")
digit <- mx.symbol.broadcast_to(data=digit, shape=c(28,28,10, batch_size), name="digit_broadcast")

data_concat <- mx.symbol.concat(list(data, digit), num.args = 2, dim = 1, name='d_concat')

d1 <- mx.symbol.Convolution(data=data_concat, name='d1', kernel=c(5,5), stride=c(1,1), pad=c(2,2), num_filter=dis_features, no_bias=no_bias)
d1 <- mx.symbol.BatchNorm(d1, name='dbn1', fix_gamma=fix_gamma, eps=eps)
d1 <- mx.symbol.relu(d1, name='dact1')
d1 <- mx.symbol.Pooling(data=d1, name="pool1", pool_type="max", kernel=c(2,2), stride=c(2,2), pad=c(0,0))

d2 <- mx.symbol.Convolution(d1, name='d2', kernel=c(3,3), stride=c(1,1), pad=c(1,1), num_filter=dis_features*2, no_bias=no_bias)
d2 <- mx.symbol.BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
d2 <- mx.symbol.relu(d2)
d2 <- mx.symbol.Pooling(data=d2, pool_type="max", kernel=c(2,2), stride=c(2,2), pad=c(0,0))

d3 <- mx.symbol.Convolution(d2, name='d3', kernel=c(3,3), stride=c(1,1), pad=c(0,0), num_filter=dis_features*2, no_bias=no_bias)
dbn3 <- mx.symbol.BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
dact3 <- mx.symbol.relu(dbn3, name='dact3')

d4 <- mx.symbol.Convolution(dact3, name='d4', kernel=c(5,5), stride=c(1,1), pad=c(0,0), num_filter=dis_features*4, no_bias=no_bias)
dbn4 <- mx.symbol.BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=eps)
dact4 <- mx.symbol.relu(dbn4, name='dact4')

dflat <- mx.symbol.Flatten(dact4, name="dflat")

dfc <- mx.symbol.FullyConnected(data=dflat, name="dfc", num_hidden=1, no_bias=F)
# D_sym <- mx.symbol.LogisticRegressionOutput(data=dfc, label=real_flag, name='D_sym')
# D_sym <- mx.symbol.LinearRegressionOutput(data=dfc, label=real_flag, name='D_sym')
D_sym <- mx.symbol.MAERegressionOutput(data=dfc, label=real_flag, name='D_sym')


########################
### Graph
########################
input_shape_D <- c(28, 28, 1, batch_size)

graph.viz(G_sym, direction = "LR", shape=list(label = batch_size))
graph.viz(D_sym, direction = "LR", shape=list(data = input_shape_D, digit = batch_size))


#######################
### Training
#######################

iter <- mx.io.arrayiter(data = train_data, label = train_label, batch.size = batch_size, shuffle = T)
initializer <- mx.init.Xavier(rnd_type = "gaussian", factor_type = "avg", magnitude = 3)

# adadelta
optimizer_G <- mxnet:::mx.opt.adadelta(rho = 0.9, epsilon = 1e-8, wd = 0, rescale.grad = 1/batch_size, clip_gradient = 1)
optimizer_D <- mxnet:::mx.opt.adadelta(rho = 0.9, epsilon = 1e-8, wd = 0, rescale.grad = 1/batch_size, clip_gradient = 1)

# rmsprop
# optimizer_G <- mxnet:::mx.opt.rmsprop(learning.rate = 1e-4, gamma1 = 0.95, gamma2 = 0.9, epsilon = 1e-5, wd = 1e-8, rescale.grad = 1/batch_size, clip_gradient = 1)
# optimizer_D <- mxnet:::mx.opt.rmsprop(learning.rate = 1e-5, gamma1 = 0.95, gamma2 = 0.9, epsilon = 1e-5, wd = 1e-8, rescale.grad = 1/batch_size, clip_gradient = 0.1)

# adam
# optimizer_G <- mxnet:::mx.opt.adam(learning.rate = 1e-4, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, wd = 0, rescale.grad = 1/batch_size, clip_gradient = 1)
# optimizer_D <- mxnet:::mx.opt.adam(learning.rate = 1e-4, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, wd = 0, rescale.grad = 1/batch_size, clip_gradient = 1)

iter$reset()
iter$iter.next()

arguments_G <- G_sym$arguments
arguments_D <- D_sym$arguments

input_names_G <- intersect(names(iter$value()), arguments_G)
input_names_D <- c(intersect(names(iter$value()), arguments_D))

input.shape.G <- sapply(input_names_G, function(n) {
  dim(iter$value()[[n]])
}, simplify = FALSE)

input.shape.D <- sapply(input_names_D, function(n) {
  dim(iter$value()[[n]])
}, simplify = FALSE)

shapes_G <- G_sym$infer.shape(input.shape.G)
shapes_D <- D_sym$infer.shape(c(input.shape.D, digit = batch_size))

# initialize all arguments with zeros
arg.params.G <- lapply(shapes_G$arg.shapes, function(shape) {
  mx.nd.zeros(shape = shape, ctx = mx.cpu())
})

arg.params.D <- lapply(shapes_D$arg.shapes, function(shape) {
  mx.nd.zeros(shape = shape, ctx = mx.cpu())
})

# initialize input parameters
input_params_G <- arg.params.G[input_names_G]
input_params_D <- arg.params.D[input_names_D]

# initialize parameters - only argument ending with _weight and _bias are initialized
arg.params.ini.G <- mx.init.create(initializer = initializer, shape.array = shapes_G$arg.shapes, ctx = mx.cpu(), skip.unknown = TRUE)
arg.params.ini.D <- mx.init.create(initializer = initializer, shape.array = shapes_D$arg.shapes, ctx = mx.cpu(), skip.unknown = TRUE)

# assign initilaized parameters to arg.params
arg.params.G[names(arg.params.ini.G)] <- arg.params.ini.G
arg.params.D[names(arg.params.ini.D)] <- arg.params.ini.D

# remove input params from arg.params
arg.params.G[input_names_G] <- NULL
arg.params.D[input_names_D] <- NULL

# arg.params to be updated
arg.params.names.G <- names(arg.params.G)
arg.params.names.D <- names(arg.params.D)

# Grad request - null: no gradient to compute write: write gradient to execuotr
grad.req.G <- rep("null", length(arguments_G))
grad.req.write.G <- arguments_G %in% arg.params.names.G
grad.req.G[grad.req.write.G] <- "write"

# for discriminator, we want to get the gradient for the data as it will serve as input gradient for the generator
grad.req.D <- rep("null", length(arguments_D))
grad.req.write.D <- arguments_D %in% setdiff(c(arg.params.names.D, "data"), c("real_flag", "digit"))
grad.req.D[grad.req.write.D] <- "write"

# Arg array order
update_names_G <- c(input_names_G, arg.params.names.G)
arg.update.idx.G <- match(arguments_G, update_names_G)

update_names_D <- c(input_names_D, arg.params.names.D)
arg.update.idx.D <- match(arguments_D, update_names_D)

# aux parameters setup
aux.params.G <- lapply(shapes_G$aux.shapes, function(shape) {
  mx.nd.zeros(shape = shape, ctx = mx.cpu())
})

aux.params.D <- lapply(shapes_D$aux.shapes, function(shape) {
  mx.nd.zeros(shape = shape, ctx = mx.cpu())
})

aux.params.ini.G <- mx.init.create(initializer, shapes_G$aux.shapes, ctx = mx.cpu(), skip.unknown = FALSE)
aux.params.ini.D <- mx.init.create(initializer, shapes_D$aux.shapes, ctx = mx.cpu(), skip.unknown = FALSE)

if (length(aux.params.G) > 0) {
  aux.params.G[names(aux.params.ini.G)] <- aux.params.ini.G
} else aux.params.G <- NULL

if (length(aux.params.D) > 0) {
  aux.params.G[names(aux.params.ini.D)] <- aux.params.ini.D
} else aux.params.D <- NULL


####################################################################
#### Training Loop
mx.symbol.bind <- mxnet:::mx.symbol.bind
ctx <- mx.gpu()
message("Start training")

input_G <- input_params_G
input_D <- input_params_D

# initialize executors
exec_G <- mx.symbol.bind(symbol = G_sym, arg.arrays = c(input_G, arg.params.G)[arg.update.idx.G],
                         aux.arrays = aux.params.G, ctx = ctx, grad.req = grad.req.G)

exec_D <- mx.symbol.bind(symbol = D_sym, arg.arrays = c(input_D, arg.params.D)[arg.update.idx.D],
                         aux.arrays = aux.params.D, ctx = ctx, grad.req = grad.req.D)

updater_G <- mx.opt.get.updater(optimizer_G, exec_G$ref.arg.arrays, ctx = ctx)
updater_D <- mx.opt.get.updater(optimizer_D, exec_D$ref.arg.arrays, ctx = ctx)


# define a custom evaluation metric
metric_D <- mx.metric.custom("accuracy", function(label, pred) {
  label <- mx.nd.reshape(label, shape = -1)
  pred <- mx.nd.reshape(pred, shape = -1) > 0.5
  res <- mx.nd.mean(label == pred)
  return(as.array(res))
})

metric_D_value <- metric_D$init()

epoch <- 1
iteration <- 1
iter$reset()

# train loop
for (iteration in 1:5000) {
  
  if (!iter$iter.next()) {
    message(paste0("Epoch ", epoch, " completed"))
    epoch <- epoch + 1
    iter$reset()
    iter$iter.next()
  }
  
  ### Feed Generator with random labels of digits
  iter_value <- iter$value()
  mx.exec.update.arg.arrays(exec_G, arg.arrays = list(label = iter_value$label), match.name = TRUE)
  mx.exec.forward(exec_G, is.train=T)
  
  ### Feed the Discriminator with the image produced by the Generator and 
  ### Train loop on fake - real_flag is set to 0
  real_flag <- mx.nd.zeros(shape = batch_size, ctx = mx.cpu())
  mx.exec.update.arg.arrays(exec_D, 
                            arg.arrays = list(data = exec_G$ref.outputs$G_sym_output, 
                                              digit = iter_value$label, 
                                              real_flag = real_flag), 
                            match.name=TRUE)
  mx.exec.forward(exec_D, is.train = T)
  preds <- mx.nd.copyto(exec_D$outputs[[1]], ctx = mx.cpu())
  mx.exec.backward(exec_D)
  
  update_args_D <- updater_D(weight = exec_D$ref.arg.arrays, grad = exec_D$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec = exec_D, arg.arrays = update_args_D, skip.null=TRUE)
  
  metric_D_value <- metric_D$update(label = real_flag, 
                                    pred = preds, 
                                    state = metric_D_value)
  
  ### Train loop on real
  mx.exec.update.arg.arrays(exec_D, 
                            arg.arrays = list(data = iter_value$data, 
                                              digit = iter_value$label, 
                                              real_flag = mx.nd.ones(shape = batch_size)), 
                            match.name=TRUE)
  mx.exec.forward(exec_D, is.train=T)
  
  preds <- mx.nd.copyto(exec_D$ref.outputs[[1]], mx.cpu())
  labels <- mx.nd.ones(shape = batch_size, ctx = mx.cpu())
  
  mx.exec.backward(exec_D)
  update_args_D <- updater_D(weight = exec_D$ref.arg.arrays, grad = exec_D$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec = exec_D, arg.arrays = update_args_D, skip.null=TRUE)
  
  metric_D_value <- metric_D$update(label = labels,
                                    pred = preds, 
                                    state = metric_D_value)
  
  ### Update Generator weights - use discriminator data gradient as input to the generator backpropagation
  mx.exec.update.arg.arrays(exec_D, arg.arrays = list(data = exec_G$ref.outputs$G_sym_output, digit = iter_value$label, real_flag = mx.nd.ones(shape = batch_size)), match.name=TRUE)
  mx.exec.forward(exec_D, is.train=T)
  mx.exec.backward(exec_D)
  D_grads <- exec_D$ref.grad.arrays$data
  mx.exec.backward(exec_G, out_grads = D_grads)
  
  update_args_G <- updater_G(weight = exec_G$ref.arg.arrays, grad = exec_G$ref.grad.arrays)
  mx.exec.update.arg.arrays(exec_G, update_args_G, skip.null=TRUE)
  
  ### Update metrics
  #metric_G_value <- metric_G$update(values[[label_name]], exec_G$ref.outputs[[output_name]], metric_G_value)
  
  if (iteration %% 25==0){
    D_metric_result <- metric_D$get(metric_D_value)
    cat(paste0("[", iteration, "] ", D_metric_result$name, ": ", D_metric_result$value, "\n"))
  }
  
  if (iteration == 1 | iteration %% 100==0){
    
    metric_D_value<- metric_D$init()
    
    par(mfrow=c(3,3), mar=c(0.1,0.1,0.1,0.1))
    for (i in 1:9) {
      img <- as.array(exec_G$ref.outputs$G_sym_output)[,,,i]
      plot(as.cimg(img), axes=F)
    }
    
    print(as.numeric(as.array(iter_value$label))[1:9])
    
  }
}



###################################
# To save and reload the Generator
###################################
# mx.symbol.save(D_sym, filename = "models/D_sym_model_v1.json")
# mx.nd.save(exec_D$arg.arrays, filename = "models/D_aux_params_v1.params")
# mx.nd.save(exec_D$aux.arrays, filename = "models/D_aux_params_v1.params")

# mx.symbol.save(G_sym, filename = "models/G_sym_model_v1.json")
# mx.nd.save(exec_G$arg.arrays, filename = "models/G_arg_params_v1.params")
# mx.nd.save(exec_G$aux.arrays, filename = "models/G_aux_params_v1.params")

# G_sym<- mx.symbol.load("models/G_sym_model_v1.json")
# G_arg_params<- mx.nd.load("models/G_arg_params_v1.params")
# G_aux_params<- mx.nd.load("models/G_aux_params_v1.params")

# exec_G <- mx.simple.bind(symbol = G_sym, label = input.shape.G$label, ctx = ctx, grad.req = "null")
# mx.exec.update.arg.arrays(exec_G, exec_G$arg.arrays, match.name=TRUE)
# mx.exec.update.aux.arrays(exec_G, exec_G$aux.arrays, match.name=TRUE)

###################################
# Generate fake number on demand
###################################
label <- rep(0, batch_size)
label[1:9] <- 1:9
label <- mx.nd.array(label)
mx.exec.update.arg.arrays(exec_G, list(label=label), match.name=TRUE)

mx.exec.forward(exec_G, is.train=F)
par(mfrow=c(3,3), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:9) {
  img <- as.array(exec_G$ref.outputs$G_sym_output)[,,,i]
  plot(as.cimg(img), axes=F)
}
