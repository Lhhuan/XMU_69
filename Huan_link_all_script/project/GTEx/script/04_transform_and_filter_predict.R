# library(ggplot2)
library(Rcpp)
library(readxl)
library(dplyr)
library(stringr)
library(ggpubr)
library(gridExtra)
library(data.table)
# library(Seurat)
# library(clusterProfiler)
# library(org.Hs.eg.db)
library(readr)
# library(EnsDb.Hsapiens.v79)
library(parallel)

setwd("/public/home/huanhuan/project/GTEx/output/")
# org <- read_tsv("03_smiliarity.txt",col_names =T)%>%as.data.frame()
org <- fread("03_smiliarity.txt",header =T,sep="\t")%>%as.data.frame()

# org1 <- org[which(org>0.098324805)]

# f = mclapply(1:ncol(org), function(i){
#   cat(i,"\n")
#   x = org[,i]
#   ind = which(x>0.098324805)

f = mclapply(1:ncol(org), function(i){
  cat(i,"\n")
  x = org[,i]
  ind = which(x>0.098324805)
  value = x[ind]
  data.frame(hotspot_id = ind-1, 
             gene_id = i + 122368,
             smiliarity = value,
             stringsAsFactors = FALSE)
}, mc.cores = 1)

aaa <-do.call(rbind,f)

write.table(aaa,"04_transform_and_filter_predict.txt",col.names=T,row.names =F,quote=F,sep="\t")





# f = mclapply(1:nrow(org), function(i){
#   cat(i,"\n")
#   x = org[,i]
#   ind = which(x>0.098324805)
#   mclapply(ind,function(j=NULL){
#       data.frame(hotspot_id = j-1,
#         gene_id=122368 +i,
#         value=x[j],
#         stringsAsFactors=FALSE)
#   },mc.cores=2)
# },mc.cores=1)

# final_t <-do.call(rbind,f)

# org <- fread("03_smiliarity.txt",header =T,sep="\t")%>%as.data.frame()


# org<-read.table("03_smiliarity.txt",header =F,sep="\t")%>%data.frame()