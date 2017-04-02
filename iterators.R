require(mxnet)

G_iterator<- function(batch_size){
  
  batch<- 0
  batch_per_epoch<-5
  
  reset<- function(){
    #set.seed(123)
    batch<<- 0
    ### Return first BucketID at reset for initialization of the model
    #bucketID<<- bucket_plan[1]
  }
  
  iter.next<- function(){
    batch<<- batch+1
    #set.seed(123)
    if (batch>batch_per_epoch) {
      return(FALSE)
    } else {
      return(TRUE)
    }
  }
  
  value<- function(){
    #set.seed(123+batch)
    digit<- sample(10, size = batch_size, replace = T)
    data<- array(0, dim = c(10, batch_size))
    data[cbind(digit, 1:batch_size)]<-1
    dim(data)<- c(1, 1, 10, batch_size)
    return(list(data=mx.nd.array(data), digit=mx.nd.array(digit-1)))
  }
  
  return(list(reset=reset, iter.next=iter.next, value=value, batch_size=batch_size, batch=batch))
}

D_iterator<- function(batch_size){
  
  batch<- 0
  batch_per_epoch<-5
  
  reset<- function(){
    #set.seed(123)
    batch<<- 0
    ### Return first BucketID at reset for initialization of the model
    #bucketID<<- bucket_plan[1]
  }
  
  iter.next<- function(){
    batch<<- batch+1
    #set.seed(123)
    if (batch>batch_per_epoch) {
      return(FALSE)
    } else {
      return(TRUE)
    }
  }
  
  value<- function(){
    #set.seed(123+batch)
    idx<- sample(length(train_labels), size = batch_size, replace = T)
    label<- train_labels[idx]
    data<- train_array[,,,idx, drop=F]
    digit<- array(0, dim = c(10, batch_size))
    digit[cbind(label+1, 1:batch_size)]<-1
    
    return(list(data=mx.nd.array(data), digit=mx.nd.array(digit), label=mx.nd.array(label)))
  }
  
  return(list(reset=reset, iter.next=iter.next, value=value, batch_size=batch_size, batch=batch))
}

# 
# iter_D<- D_iterator(16)
# iter_D$reset()
# iter_D$iter.next()
# iter_D_values<- iter_D$value()
# dim(iter_D_values$data)
# dim(iter_D_values$digit)
# dim(iter_D_values$label)
# 
# iter_G<- G_iterator(16)
# iter_G$reset()
# iter_G$iter.next()
# iter_G_values<- iter_G$value()
# 
# dim(iter_G_values$data)
# dim(iter_G_values$digit)
# as.array(iter_G_values$data)[1,1,,]
# 
# summary(as.array(iter_G_values$data))
# iter_G_values$digit
# iter_G$batch
# 
# class(data_test)
# dim(data_test)
# data_test
# iter_G$value()$label
# iter_G$batch
