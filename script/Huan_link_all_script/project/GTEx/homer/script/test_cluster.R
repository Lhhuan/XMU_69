# install_github(renjun0324/KKLClustering)
# library(irlba)
library(data.table)
# library(KKLClustering)
library(dplyr)
# data = fread("/share/swap/huan/6mers.csv",data.table=FALSE)
setwd("/public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer/")

dd <-function(i){
    C = fread(paste0("communities_",i,".bed.gz"), data.table=FALSE)
    C0 = filter(C,V4==0)%>%select(V1,V2,V3)
    C1 = filter(C,V4==1)
    write.table(C0,paste0("./test/C0/",i,"_com0.bed"),col.names=F,row.names=F,quote=F,sep="\t")
    write.table(C1,paste0("./test/C1/",i,"_com1.bed"),col.names=F,row.names=F,quote=F,sep="\t")
}


lapply(c(3:9),dd)

