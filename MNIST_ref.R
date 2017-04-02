require("data.table")
require("dplyr")
require("tidyr")
require("readr")
require("ggplot2")
require("plotly")
require("DiagrammeR")
require("DT")
require("mxnet")


#############################################

train <- read.csv('data/train.csv', header=TRUE)
test <- read.csv('data/test.csv', header=TRUE)

set.seed(123123)
eval_rows<- sample(1:nrow(train), size = round(0.1*nrow(train),0), replace = F)
eval<- train[eval_rows,]
train<- train[-eval_rows,]


train <- data.matrix(train)
eval <- data.matrix(eval)
test <- data.matrix(test)

train_x <- train[,-1]
train_labels <- train[,1]

eval_x <- eval[,-1]
eval_labels <- eval[,1]

train_x <- t(train_x/255)
eval_x <- t(eval_x/255)
test <- t(test/255)


################################################################
#### Convert data to arrays
train.array <- train_x
dim(train.array) <- c(28, 28, 1, ncol(train_x))
eval.array <- eval_x
dim(eval.array) <- c(28, 28, 1, ncol(eval_x))
test.array <- test
dim(test.array) <- c(28, 28, 1, ncol(test))

##################################################
#### Conv Net
##################################################

# input
data <- mx.symbol.Variable('data')

rand<- mx.symbol.normal(loc=0., scale=0.01, shape=c(28,28,1,128), name="rand")
data_noise<- mx.symbol.elemwise_add(data,rand, name="data_noise")

conv0 <- mx.symbol.Convolution(data=data_noise, kernel=c(3,3), num_filter=16, stride=c(1,1), pad=c(1,1))
bn0 <- mx.symbol.BatchNorm(data=conv0)
#act0 <- mx.symbol.LeakyReLU(data=bn0, act_type="elu",  slope=0.25)
act0 <- mx.symbol.Activation(data=bn0, act_type="relu")
pool0 <- mx.symbol.Pooling(data=act0, pool_type="max", kernel=c(2,2), stride=c(2,2), pad=c(0,0))

conv1 <- mx.symbol.Convolution(data=pool0, kernel=c(3,3), num_filter=24, stride=c(1,1), pad=c(1,1))
bn1 <- mx.symbol.BatchNorm(data=conv1)
#act1 <- mx.symbol.LeakyReLU(data=bn1, act_type="elu", slope=0.25)
act1 <- mx.symbol.Activation(data=bn1, act_type="relu")
pool1 <- mx.symbol.Pooling(data=act1, pool_type="max", kernel=c(2,2), stride=c(2,2), pad=c(0,0))

conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(3,3), num_filter=32, stride=c(1,1), pad=c(0,0))
bn2 <- mx.symbol.BatchNorm(data=conv2)
#act2 <- mx.symbol.LeakyReLU(data=bn2, act_type="elu", slope=0.25)
act2 <- mx.symbol.Activation(data=bn2, act_type="relu")
#pool2 <- mx.symbol.Pooling(data=act2, pool_type="max", kernel=c(2,2), stride=c(2,2), pad=c(0,0))

conv3 <- mx.symbol.Convolution(data=act2, kernel=c(5,5), num_filter=48, stride=c(1,1), pad=c(0,0))
bn3 <- mx.symbol.BatchNorm(data=conv3)
#act3 <- mx.symbol.LeakyReLU(data=bn3, act_type="elu", slope=0.25)
act3 <- mx.symbol.Activation(data=bn3, act_type="relu")
#pool3 <- mx.symbol.Pooling(data=act3, pool_type="avg", kernel=c(7,7), stride=c(2,2), pad=c(0,0))

flatten <- mx.symbol.Flatten(data=act3)
drop_flat <- mx.symbol.Dropout(data=flatten, p=0.25)

# second fullc
fc2 <- mx.symbol.FullyConnected(data=drop_flat, num_hidden=10)

# loss
convnet <- mx.symbol.SoftmaxOutput(data=fc2)
devices <- mx.cpu()

graph.viz(convnet, shape = c(28,28,1,128), type = "vis", direction = "UD")

optimizer<- mx.opt.create(name = "adadelta",
                          rho=0.92,
                          epsilon=5e-6,
                          wd=0.0005,
                          clip_gradient=NULL,
                          rescale.grad=1/128)

mx.set.seed(123)
system.time(
  model_convnet_1 <- mx.model.FeedForward.create(convnet, X=train.array, y=train_labels, eval.data = list(data=eval.array, label=eval_labels), 
                                                 ctx=devices, num.round=16, array.batch.size=128,
                                                 optimizer=optimizer,
                                                 eval.metric=mx.metric.accuracy, 
                                                 initializer=mx.init.Xavier(rnd_type = "gaussian", factor_type = "avg", magnitude = 3),
                                                 batch.end.callback=mx.callback.log.train.metric(50),
                                                 epoch.end.callback=mx.callback.log.train.metric(1))
)


predict_eval<- t(predict(model = model_convnet_1, X = eval.array))
predict_labels<- max.col(predict_eval)-1
