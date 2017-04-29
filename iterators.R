
G_iterator<- function(batch_size){
  
  batch<- 0
  batch_per_epoch<-5
  
  reset<- function(){
    batch<<- 0
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
    set.seed(123+batch)
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
    batch<<- 0
  }
  
  iter.next<- function(){
    batch<<- batch+1
    if (batch>batch_per_epoch) {
      return(FALSE)
    } else {
      return(TRUE)
    }
  }
  
  value<- function(){
    set.seed(123+batch)
    idx<- sample(length(train_labels), size = batch_size, replace = T)
    label<- train_labels[idx]
    data<- train_array[,,,idx, drop=F]
    digit<- array(0, dim = c(10, batch_size))
    digit[cbind(label+1, 1:batch_size)]<-1
    
    return(list(data=mx.nd.array(data), digit=mx.nd.array(digit), label=mx.nd.array(label)))
  }
  
  return(list(reset=reset, iter.next=iter.next, value=value, batch_size=batch_size, batch=batch))
}


