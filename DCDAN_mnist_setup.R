require("data.table")
require("dplyr")
require("tidyr")
require("readr")
require("ggplot2")
require("plotly")
require("DiagrammeR")
require("DT")
require("imager")
require("mxnet")


#############################################
train <- read.csv('data/train.csv', header=TRUE)
test <- read.csv('data/test.csv', header=TRUE)

set.seed(123)
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

train_x <- t(train_x/255*2-1)
eval_x <- t(eval_x/255*2-1)
test <- t(test/255*2-1)


################################################################
#### Convert data to arrays
train_array <- train_x
dim(train_array) <- c(28, 28, 1, ncol(train_x))
eval_array <- eval_x
dim(eval_array) <- c(28, 28, 1, ncol(eval_x))
test_array <- test
dim(test_array) <- c(28, 28, 1, ncol(test))

##################################################
#### Generator Symbol
##################################################
random_dim<- 128
gen_features<- 64
dis_features<- 24
image_depth = 1
fix_gamma<- T
no_bias<- T
eps<-1e-5 + 1e-12

batch_size<- 64

data = mx.symbol.Variable('data')

gen_rand<- mx.symbol.normal(loc=0, scale=1, shape=c(1, 1, random_dim, batch_size), name="gen_rand")
gen_concat<- mx.symbol.Concat(data = list(data, gen_rand), num.args = 2, name="gen_concat")

g1 = mx.symbol.Deconvolution(gen_concat, name='g1', kernel=c(4,4), num_filter=gen_features*4, no_bias=T)
gbn1 = mx.symbol.BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=eps)
gact1 = mx.symbol.Activation(gbn1, name='gact1', act_type='relu')

g2 = mx.symbol.Deconvolution(gact1, name='g2', kernel=c(3,3), stride=c(2,2), pad=c(1,1), num_filter=gen_features*2, no_bias=no_bias)
gbn2 = mx.symbol.BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=eps)
gact2 = mx.symbol.Activation(gbn2, name='gact2', act_type='relu')

g3 = mx.symbol.Deconvolution(gact2, name='g3', kernel=c(4,4), stride=c(2,2), pad=c(1,1), num_filter=gen_features, no_bias=no_bias)
gbn3 = mx.symbol.BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=eps)
gact3 = mx.symbol.Activation(gbn3, name='gact3', act_type='relu')

# g4 = mx.symbol.Deconvolution(gact3, name='g4', kernel=c(4,4), stride=c(2,2), pad=c(1,1), num_filter=ngf, no_bias=no_bias)
# gbn4 = mx.symbol.BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=eps)
# gact4 = mx.symbol.Activation(gbn4, name='gact4', act_type='relu')

g5 = mx.symbol.Deconvolution(gact3, name='g5', kernel=c(4,4), stride=c(2,2), pad=c(1,1), num_filter=image_depth, no_bias=no_bias)
G_sym = mx.symbol.Activation(g5, name='gact5', act_type='tanh')


##################################################
#### Discriminator Symbol
##################################################
data = mx.symbol.Variable('data')
dis_digit = mx.symbol.Variable('digit')
label = mx.symbol.Variable('label')

d1 = mx.symbol.Convolution(data, name='d1', kernel=c(4,4), stride=c(2,2), pad=c(1,1), num_filter=24, no_bias=no_bias)
#dbn1 = mx.symbol.BatchNorm(d1, name='dbn1', fix_gamma=fix_gamma, eps=eps)
dact1 = mx.symbol.LeakyReLU(d1, name='dact1', act_type='elu', slope=0.25)

d2 = mx.symbol.Convolution(dact1, name='d2', kernel=c(4,4), stride=c(2,2), pad=c(1,1), num_filter=32, no_bias=no_bias)
dbn2 = mx.symbol.BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
dact2 = mx.symbol.LeakyReLU(dbn2, name='dact2', act_type='elu', slope=0.25)

# d3 = mx.symbol.Convolution(dact2, name='d3', kernel=c(4,4), stride=c(2,2), pad=c(1,1), num_filter=ndf*4, no_bias=no_bias)
# dbn3 = mx.symbol.BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
# dact3 = mx.symbol.LeakyReLU(dbn3, name='dact3', act_type='elu', slope=0.25)

d4 = mx.symbol.Convolution(dact2, name='d4', kernel=c(3,3), stride=c(1,1), pad=c(0,0), num_filter=48, no_bias=no_bias)
dbn4 = mx.symbol.BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=eps)
dact4 = mx.symbol.LeakyReLU(dbn4, name='dact4', act_type='elu', slope=0.25)

d5 = mx.symbol.Convolution(dact4, name='d5', kernel=c(5,5), pad=c(0,0), num_filter=64, no_bias=no_bias)
dflat = mx.symbol.Flatten(d5, name="dflat")

dflat_concat <- mx.symbol.Concat(list(dflat, dis_digit), num.args = 2, dim = 1, name='dflat_concat')

dfc1 <- mx.symbol.FullyConnected(data=dflat_concat, name="dfc1", num_hidden=32, no_bias=F)
dfc1_act<- mx.symbol.LeakyReLU(dfc1, name='dfc1_act', act_type='elu', slope=0.25)

dfc <- mx.symbol.FullyConnected(data=dfc1_act, name="dfc", num_hidden=1, no_bias=F)
D_sym = mx.symbol.LogisticRegressionOutput(data=dfc, label=label, name='dloss')


########################
### Graph
########################
input_shape_G<- c(1, 1, 10, batch_size)
input_shape_D<- c(28, 28, 1, batch_size)

graph.viz(G_sym, type = "vis", direction = "UD", shape=input_shape_G)
graph.viz(D_sym, type = "vis", direction = "UD", shape=input_shape_D)

