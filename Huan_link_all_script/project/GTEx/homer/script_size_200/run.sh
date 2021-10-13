scp huanhuan@59.77.18.162:/share/data0/QTLbase/huan/GTEx/Tissue_merge/Cis_eQTL/hotspot/interval_18/chr1_11/kmer/6/communities_*.bed.gz /public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer/
source activate huan_py3

perl 11_split_cluster_and_homer_other_n_community.pl

perl 11_split_cluster_and_homer_other_6_community.pl
perl 11_split_cluster_and_homer_other_7_community.pl
perl 11_split_cluster_and_homer_other_8_community.pl
perl 11_split_cluster_and_homer_other_9_community.pl
perl 11_split_cluster_and_homer_other_6_community.pl


scp -r /public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer/*_community/ huanhuan@59.77.18.162:/share/data0/QTLbase/huan/GTEx/Tissue_merge/Cis_eQTL/hotspot/interval_18/chr1_11/kmer/6/

cd /public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer/5_community/homer/
less communities_2.bed communities_3.bed >communities_2_3.bed

mkdir 2_3
findMotifsGenome.pl communities_2_3.bed  hg19 /public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer/5_community/homer/2_3 -size 200 
scp -r /public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer/5_community/homer/2_3 huanhuan@59.77.18.162:/share/data0/QTLbase/huan/GTEx/Tissue_merge/Cis_eQTL/hotspot/interval_18/chr1_11/kmer/6/5_community/homer/



# findMotifsGenome.pl "/public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer/7_community/homer/communities_1.bed" hg19 /public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer/7_community/homer/1/ -size 200




# cp -r  6kmer 6kmer_muliti

# findMotifsGenome.pl "/public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer_muliti/5_community/homer/communities_2.bed" hg19 /public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer_muliti/5_community/homer/test2_refine/ -size 200


# findMotifsGenome.pl "/public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer_muliti/5_community/homer/communities_2.bed" hg19 /public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer_muliti/5_community/homer/test2/ -size 200

# cp -r /public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer_muliti/5_community/homer/test2/ /public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer_muliti/5_community/homer/test2_2/

# findMotifsGenome.pl "/public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer_muliti/5_community/homer/communities_2.bed" hg19 /public/home/huanhuan/project/GTEx/homer/output/chr1_11/6kmer_muliti/5_community/homer/test2/ -size 200


# perl 11_split_cluster_and_homer_other_6_community.pl