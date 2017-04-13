require("mxnet")

xx1<- mx.nd.array(array(c(0,1,0,0), dim=4))
xx2<- mx.nd.Reshape(data=xx1, shape = c(1,1,4))
dim(xx2)
xx3<- mx.nd.broadcast.to(data=xx2, shape = c(2,2,4))
dim(xx3)
