library(ggplot2)
library(sqldf)
library(plyr)
library(dplyr)
 
input_directory  <-  '/Users/kdo210/Documents/capita/SET2'

magnitude = 1000000

big_50 = read.table(paste(input_directory,"/results_new50_2/set2_mlp_srelu_sgd_cifar10_params.txt", sep=''))
big_10 = read.table(paste(input_directory,"/results_new10_2/set2_mlp_srelu_sgd_cifar10_params.txt", sep=''))
small_10 = read.table(paste(input_directory,"/results_final/set2_mlp_srelu_sgd_cifar10_params.txt", sep=''))

big_50[,"V1"] <- round(big_50[,"V1"]/magnitude,0)
big_10[,"V1"] <- round(big_10[,"V1"]/magnitude,0)
small_10[,"V1"] <- round(small_10[,"V1"]/magnitude,0)

big_50[,"params_big50"] <- cumsum(big_50$V1)
big_10[,"params_big10"] <- cumsum(big_10$V1)
small_10[,"params_small10"] <- cumsum(small_10$V1)

params_dense = 20337010/magnitude
params_fix = (141659+99568+99833+40010)/magnitude
params_set = params_fix

dense = data.frame(rep(params_dense, 1000))
names(dense) <- 'V1'
dense[,"params_dense"] <- round(cumsum(dense$V1),0)
dense[,"params_dense"] <- dense[,"params_dense"] 

fix = data.frame(rep(params_fix, 1000))
names(fix) <- 'V1'
fix[,"params_fix"] <- round(cumsum(fix$V1),0)
fix[,"params_fix"] <- fix[,"params_fix"]

set = data.frame(rep(params_set, 1000))
names(set) <- 'V1'
set[,"params_set"] <- cumsum(set$V1)
set[,"params_set"] <- round(set[,"params_set"],0)

epochs = data.frame(seq(1, 1000, 1))
names(epochs) = 'epochs'

all_params = cbind(
   'Epochs'= epochs,
  'Dense'=dense$params_dense,
  'Fix'=fix$params_fix ,
  'SET'=set$params_set,
  'SPET_lb_20' =big_50$params_big50,
  'SPET_lb_100'= big_10$params_big10,
  'SPET_sb_100' = small_10$params_small10
  )

all_params_abs = cbind(
  'Epochs'= epochs,
  'Dense'=dense$V1,
  'Fix'=fix$V1 ,
  'SET'=set$V1,
  'SPET_lb_20' =big_50$V1,
  'SPET_lb_100'= big_10$V1,
  'SPET_sb_100' = small_10$V1
)

graph <- ggplot(data=all_params, aes(x=epochs))
graph = graph + geom_line(aes_string(y='Dense', colour=shQuote('Dense') ), size=2)
graph = graph + geom_line(aes_string(y='Fix', colour=shQuote('Fix') ), size=2)
graph = graph + geom_line(aes_string(y='SET', colour=shQuote('SET') ), size=2)
graph = graph + geom_line(aes_string(y='SPET_lb_20', colour=shQuote('SPET_lb_20') ), size=2)
graph = graph + geom_line(aes_string(y='SPET_lb_100', colour=shQuote('SPET_lb_100') ), size=2)
graph = graph + geom_line(aes_string(y='SPET_sb_100', colour=shQuote('SPET_sb_100') ), size=2)
graph = graph  + theme(legend.position="bottom" , 
                       legend.text=element_text(size=15), 
                       axis.text=element_text(size=25),axis.title=element_text(size=30),
                       plot.subtitle=element_text(size=20 )) 
graph = graph  + labs( y="Accumulated Params (millions)", x="Epochs" ) 

 
ggsave(paste( input_directory,'/params.pdf',  sep=''), graph , device='pdf', height = 8, width = 8)


 

graph <- ggplot(data=all_params_abs, aes(x=epochs))
graph = graph + geom_line(aes_string(y='Dense', colour=shQuote('Dense') ), size=2)
graph = graph + geom_line(aes_string(y='Fix', colour=shQuote('Fix') ), size=2)
graph = graph + geom_line(aes_string(y='SET', colour=shQuote('SET') ), size=2)
graph = graph + geom_line(aes_string(y='SPET_lb_20', colour=shQuote('SPET_lb_20') ), size=2)
graph = graph + geom_line(aes_string(y='SPET_lb_100', colour=shQuote('SPET_lb_100') ), size=2)
graph = graph + geom_line(aes_string(y='SPET_sb_100', colour=shQuote('SPET_sb_100') ), size=2)
graph = graph  + theme(legend.position="bottom" , 
                       legend.text=element_text(size=15), 
                       axis.text=element_text(size=25),axis.title=element_text(size=30),
                       plot.subtitle=element_text(size=20 )) 
graph = graph  + labs( y="Params (millions)", x="Epochs" ) 


ggsave(paste( input_directory,'/params_abs.pdf',  sep=''), graph , device='pdf', height = 8, width = 8)

