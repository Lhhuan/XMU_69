install_github(renjun0324/KKLClustering)
library(irlba)
library(data.table)
library(KKLClustering)

# data = fread("/share/swap/huan/6mers.csv",data.table=FALSE)
data = fread("/share/data0/GTEx/test_kmer/chr1_22_6kmers.csv", data.table=FALSE)
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

# install_github(renjun0324/KKLClustering)