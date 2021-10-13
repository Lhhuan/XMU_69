library(irlba)
library(data.table)
library(KKLClustering)

data = fread("/share/data0/GTEx/test_kmer/chr1_11_6mers.csv",data.table=FALSE)
# data = fread("/share/data0/GTEx/test_kmer/chr1_22_6kmers.csv", data.table=FALSE)
rownames(data) = data[,1]
data = data[,-1]

#-------------------------------------------------------------------------------
#                                                                              
#                                      PCA                                 
#                                                                              
#-------------------------------------------------------------------------------

pca = prcomp_irlba(data, n = 50)
pca_result = pca$x
rownames(pca_result) = rownames(data)

#-------------------------------------------------------------------------------
#                                                                              
#                                   clustering                                 
#                                                                              
#-------------------------------------------------------------------------------

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


#-------------------------------------------------------------------------------
#                                                                              
#                                和seekr的结果对比                                 
#                                                                              
#-------------------------------------------------------------------------------

bed <- as.data.frame(read.table("/share/data0/GTEx/test_kmer/chr1_11_communities_5.bed.gz",
								header = FALSE, sep="\t",stringsAsFactors=FALSE, quote=""))
rn <- sapply(1:nrow(bed), function(i){
	x=bed[i,]
	paste0(">",x[1],":",x[2],"-",x[3])
})
all.equal(rn, rownames(data))
rownames(bed) = rn

aricode::ARI(bed$V4, result$cluster_df$cluster)
# 0.5479268
