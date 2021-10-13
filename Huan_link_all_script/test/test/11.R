library("randomForest")
library(ggplot2)
library(Rcpp)
library(readxl)
library(dplyr)
library(caret)
library(stringr)
# library(lattice)
library(e1071) #libsvm
library(plotROC)
library(pROC)
library(parallel)
library(Metrics)

normal_feature <- c()

mean_and_sd<-function(org_tissue=NULL){
  org_median <-org_tissue %>% dplyr::select(average_effective_drug_target_score_SNV_INDEL,max_effective_drug_target_score_SNV_INDEL,average_mutation_map_to_gene_level_score_SNV_INDEL,max_mutation_map_to_gene_level_score_SNV_INDEL,average_the_shortest_path_length_SNV_INDEL,min_the_shortest_path_length_SNV_INDEL,min_rwr_normal_P_value_SNV_INDEL,median_rwr_normal_P_value_SNV_INDEL,cancer_gene_exact_match_drug_target_ratio_SNV_INDEL,average_mutation_pathogenicity_SNV_INDEL,max_mutation_pathogenicity_SNV_INDEL,max_mutation_frequency_SNV_INDEL,averge_mutation_frequency_SNV_INDEL)
  
  mean_value<- lapply(colnames(org_median),function(i){
    a <- org_median[,i]
    if(length(unique(a)) > 1){
      normal_feature <- append(normal_feature,colnames(org_median)[1])
      b <-mean(a)
    }else{
      b <-0
    }
    b
  })

#-------------------
  sd_value<- lapply(colnames(org_median),function(i){
    a <- org_median[,i]
    if(length(unique(a)) > 1){
      normal_feature <- append(normal_feature,colnames(org_median)[1])
      b <-sd(a)
    }else{
      b <-0
    }
    b
  })
  training_mean<-do.call(cbind, mean_value)%>%as.data.frame()
  colnames(training_mean)<-colnames(org_median)
  training_sd<-do.call(cbind, sd_value)%>%as.data.frame()
  colnames(training_sd)<-colnames(org_median)
  train_par <-rbind(training_mean,training_sd) #第一行是mean,第二行是sd
#--------------

  train_par$drug_repurposing <-org_tissue[1:2,"AUC"] #-------------------
  #--------------------------------------------------------------------------------------
  org1$Cell_line <-org_tissue$Cell_line
  fff <-org_tissue[,41:566]
  org1 <-bind_cols(org1,fff)
  return(org1)

  training_par<-cbind(training_mean,training_sd)
  training_par<-data.frame(t(as.matrix(training_par)))
}



mean_and_sd<-function(org_tissue=NULL){
  org_median <-org_tissue %>% dplyr::select(average_effective_drug_target_score_SNV_INDEL,max_effective_drug_target_score_SNV_INDEL,average_mutation_map_to_gene_level_score_SNV_INDEL,max_mutation_map_to_gene_level_score_SNV_INDEL,average_the_shortest_path_length_SNV_INDEL,min_the_shortest_path_length_SNV_INDEL,min_rwr_normal_P_value_SNV_INDEL,median_rwr_normal_P_value_SNV_INDEL,cancer_gene_exact_match_drug_target_ratio_SNV_INDEL,average_mutation_pathogenicity_SNV_INDEL,max_mutation_pathogenicity_SNV_INDEL,max_mutation_frequency_SNV_INDEL,averge_mutation_frequency_SNV_INDEL)
  normalization<-function(x){
    return((x -mean(x)) / sd(x))} #将feature 归一化
  org_median2 <- lapply(colnames(org_median), function(i){ # i是对(colnames(org_median)进行循环
    a <- org_median[,i]
      if(length(unique(a)) > 1){
        a <- normalization(a)
      }else{
        a <- a
      }
      a
  })
  org1 <- do.call(cbind, org_median2)%>%as.data.frame()   #按列bind到一起
  colnames(org1)<-colnames(org_median)

  org1$drug_repurposing <-org_tissue$AUC
  org1$Cell_line <-org_tissue$Cell_line
  fff <-org_tissue[,41:566]
  org1 <-bind_cols(org1,fff)
  return(org1)
}


setwd("/f/mulinlab/huan/ALL_result_ICGC_ALL_drug/gene_network_merge_repurposing_model/V33/validation/Depmap_19Q4/figure/negative_sample_change/AUC/all_drug/")

load("org_gene_tissue.Rdata")
# org_all<-read.table("All_drug_gene_tissue_note.txt",header = T,sep = "\t") %>% as.data.frame()
org_all <-org_gene_tissue
set.seed(123)
random_number <-sample(x=c(1:nrow(org_all)), 40000,replace = F)
org_tissue<-org_all[random_number,]

#----------------------
org1 <-feature_extract(org_tissue=org_tissue)
Cell_line <- unique(org1$Cell_line)
cell_length <-length(Cell_line)
#-------------------

# for(random_number in c(1:10)){
cell_line_type <-data.frame()
# set.seed(18)
a <-cell_split(Cell_line=Cell_line)
for(i in c(1:length(a))){
  bbb <-Cell_line[a[[i]]] %>%as.data.frame()
  bbb$class <-i
  cell_line_type <-bind_rows(cell_line_type,bbb)
  print(i)
}

colnames(cell_line_type)[1] <-"Cell_line"
final_set <-inner_join(org1,cell_line_type,by="Cell_line")

#------cross validation 
# rs <- data.frame()
# for(i in c(1:length(a))){
#   fold_test <-filter(final_set,class ==i)
#   fold_train <-filter(final_set,class !=i)
#   # true_value1 =fold_test[,16]
#   true_value1 =fold_test$drug_repurposing
#   fold_test<-fold_test%>%dplyr::select(-drug_repurposing,-Cell_line,-class)
#   fold_train<-fold_train%>%dplyr::select(-Cell_line,-class,)
#   # fold_pre <- randomForest(drug_repurposing ~.,data=fold_train, importance=F,proximity=TRUE,type="classification")
#   fold_pre <- randomForest(drug_repurposing ~.,data=fold_train, importance=TRUE,proximity=TRUE,type="regression")
#   fold_predict <- predict(fold_pre,fold_test)
#   tmp<-data.frame(true_value1= true_value1,predict_value1=fold_predict)
#   rs <- bind_rows(rs,tmp)
#   print(i)
# }

# model_evl <-data.frame(
#   R2 = R2(rs$predict_value1, rs$true_value1),
#   rmse = rmse(rs$true_value1,rs$predict_value1),
#   mse = mse(rs$true_value1,rs$predict_value1),
#   mae = mae(rs$true_value1,rs$predict_value1)
# )

# setwd("/f/mulinlab/huan/ALL_result_ICGC_ALL_drug/gene_network_merge_repurposing_model/V33/validation/Depmap_19Q4/figure/negative_sample_change/AUC/all_drug/")
# tissue <-"All_tissue"
# # outfile <-paste(tissue,"predict.txt",sep="_")
# # print("HHH")
# # write.table(rs,outfile ,row.names = F, col.names = T,quote =F,sep="\t")
# # print("finish")

# outfile2 <-paste(tissue,"add_feature_predict_estimate.txt",sep="_")
# write.table(model_evl,outfile2 ,row.names = F, col.names = T,quote =F,sep="\t")
# print("finish")
#------------------------------------------

# for(i in c(1:length(a))){
par_cv <-function(final_set=NULL,i=NULL){
  print(i)
  fold_test <-filter(final_set,class ==i)
  fold_train <-filter(final_set,class !=i)
  # true_value1 =fold_test[,16]
  true_value1 =fold_test$drug_repurposing
  fold_test<-fold_test%>%dplyr::select(-drug_repurposing,-Cell_line,-class)
  fold_train<-fold_train%>%dplyr::select(-Cell_line,-class,)
  # fold_pre <- randomForest(drug_repurposing ~.,data=fold_train, importance=F,proximity=TRUE,type="classification")
  fold_pre <- randomForest(drug_repurposing ~.,data=fold_train, importance=TRUE,proximity=TRUE,type="regression")
  fold_predict <- predict(fold_pre,fold_test)
  tmp<-data.frame(true_value1= true_value1,predict_value1=fold_predict)
  print(i)
  return(tmp)
}
# cv_result <-mclapply(1:length(a),function(i){print(i)},mc.cores=5)
cv_result <-mclapply(1:length(a),function(i){par_cv(final_set=final_set,i=i)},mc.cores=5)
rs <-do.call(rbind,cv_result)

model_evl <-data.frame(
  R2 = R2(rs$predict_value1, rs$true_value1),
  rmse = rmse(rs$true_value1,rs$predict_value1),
  mse = mse(rs$true_value1,rs$predict_value1),
  mae = mae(rs$true_value1,rs$predict_value1)
)



setwd("/f/mulinlab/huan/ALL_result_ICGC_ALL_drug/gene_network_merge_repurposing_model/V33/validation/Depmap_19Q4/figure/negative_sample_change/AUC/all_drug/")
tissue <-"All_tissue"
# outfile <-paste(tissue,"predict.txt",sep="_")
# print("HHH")
# write.table(rs,outfile ,row.names = F, col.names = T,quote =F,sep="\t")
# print("finish")

outfile2 <-paste(tissue,"add_feature_predict_estimate_par.txt",sep="_")
outfile3 <-paste(tissue,"predict_ture_result_par.txt",sep="_")
write.table(model_evl,outfile2 ,row.names = F, col.names = T,quote =F,sep="\t")
write.table(rs,outfile3,row.names = F, col.names = T,quote =F,sep="\t")
print("finish")
