# install_github(renjun0324/KKLClustering)
library(irlba)
library(data.table)
library(KKLClustering)

# data = fread("/share/swap/huan/6mers.csv",data.table=FALSE)
data = fread("/public/home/huanhuan/project/GTEx/kmer/output/chr1_11_6kmer.csv", data.table=FALSE)
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

table(result$clsuter_df$cluster)
setwd("/public/home/huanhuan/project/GTEx/kmer/output/")
write(result$clsuter_df$cluster,"")
# install_github(renjun0324/KKLClustering)

aa <-result$cluster_df
write.table(aa,"6kmer_chr1_11_kkl.txt",col.names=T,row.names=F,quote=F,sep="\t")

pca_result1 <-as.data.frame(pca_result)
pca_result2 <-mutate(pca_result1, hotspot=rownames(pca_result), .before = 1)
colnames(pca_result2)[1]=""
write.table(pca_result2,"6kmer_chr1_11_pca.csv",col.names=T,row.names=F,quote=F,sep=",")

#--------------------------------------
bed <- as.data.frame(read.table("/public/home/huanhuan/project/GTEx/kmer/output/chr1_11_6kmer_communities_5.bed.gz",
								header = FALSE, sep="\t",stringsAsFactors=FALSE, quote=""))
rn <- sapply(1:nrow(bed), function(i){
	x=bed[i,]
	paste0(">",x[1],":",x[2],"-",x[3])
})
all.equal(rn, rownames(data))
rownames(bed) = rn

aricode::ARI(bed$V4, result$cluster_df$cluster)



#--------------------
wssplot <- function(data, nc=10, seed=1234){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)
  }
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")}
pdf("the_best_k1.pdf")
wssplot(pca_result)
dev.off()