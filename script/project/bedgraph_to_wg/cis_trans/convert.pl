
#!/usr/bin/perl
use warnings;
use strict; 
use utf8;
use List::Util qw/max min/;
use List::Util qw/sum/;

my @dirs = ("cis_10MB","cis_1MB","trans_10MB","trans_1MB");
my @number= (10..20);
foreach my $dir(@dirs){
    foreach my $i(@number){
        my $command1 =  "sort -k1,1n -k2,2n ./chr1_2/${dir}/NHPoisson_emplambda_interval_${i}cutoff_7.3_${dir}_eQTL.bedgraph >./chr1_2/${dir}/NHPoisson_emplambda_interval_${i}cutoff_7.3_${dir}_eQTL_sorted.bedgraph";
        my $command2 = "/public/home/huanhuan/tools/bedGraphToBigWig ./chr1_2/${dir}/NHPoisson_emplambda_interval_${i}cutoff_7.3_${dir}_eQTL_sorted.bedgraph  /public/home/huanhuan/reference/hg19.fa.fai  ./output_chr1_2/${dir}/NHPoisson_emplambda_interval_${i}cutoff_7.3_${dir}_eQTL.bw";
        # print "$command1\n";
        # print "$command2\n";
        system $command1;
        system $command2;
    }
}


