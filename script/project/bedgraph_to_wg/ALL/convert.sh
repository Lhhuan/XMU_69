# for((i=6;i<17;i+=1))
for((i=17;i<33;i+=1))
do
    sort -k1,1n -k2,2n ./input/NHPoisson_emplambda_interval_${i}cutoff_7.3_all_eQTL.bedgraph >./input/NHPoisson_emplambda_interval_${i}cutoff_7.3_all_eQTL_sorted.bedgraph
   /public/home/huanhuan/tools/bedGraphToBigWig ./input/NHPoisson_emplambda_interval_${i}cutoff_7.3_all_eQTL_sorted.bedgraph  /public/home/huanhuan/reference/hg19.fa.fai  ./output/NHPoisson_emplambda_interval_${i}cutoff_7.3_all_eQTL.bw 
done