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
library(circlize)
# library(Hmisc)



setwd("/public/home/huanhuan/project/GTEx/output/")
# org <- read_tsv("03_smiliarity.txt",col_names =T)%>%as.data.frame()
# org <- fread("04_transform_and_filter_predict.txt",header =T,sep="\t")%>%as.data.frame()
org <- fread("05_trans_predict.txt.gz",header =T,sep="\t")%>%as.data.frame()
org2 <-org[order(org$smiliarity,decreasing = TRUE),]





org3 <-org2%>%filter(smiliarity >0.9)
random_number <-sample(x=c(1:nrow(org3)), 50,replace = F)
org_used_fig<-org3[random_number,]



# org_used_fig <-head(org2,20 )

trans_qtl <-org_used_fig%>%select(h_chr,h_start,h_end)
trans_egene <-org_used_fig%>%select(gene_chr,gene_start,gene_end)
    pdf("predict_trans.pdf")
    par(mar = c(1, 1, 1, 1))
    circos.par(start.degree = 90)
    circos.initializeWithIdeogram(species= "hg19",chromosome.index = paste0("chr", 1:22))
    circos.genomicLink(trans_qtl, trans_egene, col = sample(nrow(trans_qtl), nrow(trans_qtl), replace = TRUE))
    # circos.genomicLink(trans_qtl, trans_egene)
    dev.off()
    circos.clear()


p_theme<-theme(panel.grid =element_blank())+theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
                                                panel.background = element_blank(), axis.title.y = element_text(size = 10),
                                                axis.title.x = element_text(size = 10),
                                                axis.line = element_line(colour = "black"))

#---------------------------------------------------------
# library(Hmisc)


# org<-read.table("/home/huanhuan/project/RNA/eQTL_associated_interaction/GTEx/output/Tissue_merge/Cis_eQTL/hotspot_cis_eQTL/interval_18/Tissue_merge_segment_hotspot_cutoff_0.176_extend_sorted_merge.bed.gz",header = F,sep = "\t") %>% as.data.frame()
# colnames(org)[1] <-"CHR"
# colnames(org)[2] <-"start"
# colnames(org)[3] <-"end"
# org$hotspot_length <-org$end - org$start
# setwd("/home/huanhuan/project/RNA/eQTL_associated_interaction/GTEx/script/Tissue_merge/figure/")
pdf("smiliarity_boxplot_distribution.pdf",width=3.5, height=3.5)

# p<-ggplot(org,aes(x=1, y=log10(hotspot_length)))+geom_violin(fill="#a3d2ca",width=0.65,outlier_colour=NA)+ theme(legend.position ="none")+p_theme+theme(axis.text.x=element_blank(),axis.ticks.x=element_blank())+xlab("Hotspot")+ylab("log10(length of hotspot)")

p<-ggplot(org,aes(x=1, y=smiliarity))+geom_violin(fill="#a3d2ca",width=0.65)+geom_boxplot(fill = "#5eaaa8",width=0.1,outlier.color=NA)+ theme(legend.position ="none")+p_theme+theme(axis.text.x=element_blank(),axis.ticks.x=element_blank())+xlab("")+ylab("smiliarity")

print(p)
dev.off()

# org <- fread("05_trans_predict.txt.gz",header =T,sep="\t")%>%as.data.frame()