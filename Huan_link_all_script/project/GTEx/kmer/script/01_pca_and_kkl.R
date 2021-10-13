# install_github(renjun0324/KKLClustering)
library(irlba)
library(data.table)
library(KKLClustering)
library(dplyr)
# data = fread("/share/swap/huan/6mers.csv",data.table=FALSE)
data = fread("/public/home/huanhuan/project/GTEx/kmer/output/chr1_22_6kmer.csv", data.table=FALSE)
rownames(data) = data[,1]
data = data[,-1]

pca = prcomp_irlba(data, n = 50)
pca_result = pca$x
rownames(pca_result) = rownames(data)

result = kkl(pca_result,
             outlier_q = 0,
             down_n = 2000,
             knn_range = seq(5,200,5),
             iter = 50,
             compute_index =  "Calinski_Harabasz",
             assess_index = "Calinski_Harabasz",
             cores = 5,
             seed = 723)

# table(result$clsuter_df$cluster)
table(result$cluster_df$cluster)
setwd("/public/home/huanhuan/project/GTEx/kmer/output/")
# write.table(result$clsuter_df,"6kmer_chr1_22_kkl.txt",col.names=T,row.names=F,quote=F,sep="\t")
aa <-result$cluster_df
write.table(aa,"6kmer_chr1_22_kkl.txt",col.names=T,row.names=F,quote=F,sep="\t")
pca_result1 <- pca_result
pca_result1 <-mutate(pca_result, hotspot=rownames(pca_result), .before = 1)

write.table(pca_result,"6kmer_chr1_22_pca.csv",col.names=T,row.names=T,quote=F,sep=",")
# install_github(renjun0324/KKLClustering)