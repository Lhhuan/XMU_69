scp huanhuan@59.77.18.162:"/home/huanhuan/project/RNA/eQTL_associated_interaction/GTEx/output/Tissue_merge/Cis_eQTL/06_merge_all_tissue_cis_sig_eQTL_hotspot_egene.txt.gz" ../data/
python 01_graph_gen_huan.py
node2
python 02_random_walk.py /public/home/huanhuan/project/GTEx/output/01_train_graph.dgl 50
python 02_random_walk.py /public/home/huanhuan/project/GTEx/output/01_test_graph.dgl 50
python 02_random_walk.py /public/home/huanhuan/project/GTEx/output/01_val_graph.dgl 50



Rscript 04_transform_and_filter_predict.R 
gzip 03_smiliarity.txt
perl 05_filter_ori_true.pl  #-------------