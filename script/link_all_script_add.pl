#!/usr/bin/perl
use warnings;
use strict;
use utf8;
use File::Basename;

my $dir_out   = "/public/home/huanhuan/Script_backup/script/";
mkdir $dir_out unless -d $dir_out;

chdir "/public/home/huanhuan/";
system "find -name *.pl > /public/home/huanhuan/perl_script";
system "find -name *.R > /public/home/huanhuan/R_script";
system "find -name *.sh > /public/home/huanhuan/all_sh_script";
system "find -name *.py > /public/home/huanhuan/python_script";
system "find -name *readme.txt > /public/home/huanhuan/readme";
system "find -name *.job > /public/home/huanhuan/job_script";

my @files = ("/public/home/huanhuan/perl_script","/public/home/huanhuan/R_script","/public/home/huanhuan/all_sh_script","/public/home/huanhuan/python_script","/public/home/huanhuan/readme","/public/home/huanhuan/job_script");
foreach my $f1(@files){
# my $f1 ="/public/home/huanhuan/perl_script";
    open my $I1, '<', $f1 or die "$0 : failed to open input file '$f1' : $!\n";
    while(<$I1>)
    {
        chomp;
        my $script = $_;
        unless($script=~/^\.\/anaconda|^\.\/Script_backup|^\.\/php|^\.\/tools|^\.\/\.local|^\.\/R|^\.\/\.conda/){  
            my $file = basename($script);
            my $dir = dirname($script);
            $dir=~s/^\.\///;
            my $do = "$dir_out/$dir";
            # mkdir  $do unless -d $do;
            unless(-e $do ){
                system "mkdir -p $do";
            }
            my $new_file ="$do/$file";
            unless(-e "$new_file"){
                #print "#$new_file\n";
                my $link1 = "ln \"$script\" \"$new_file\"" ;  #把变量引起来，这样就可以将名为12 3.pl的脚本copy 过来。而不会只copy12 而不copy 12 3.pl
            system "$link1\n";
                # print "$link1\n";
            }
        }
    }
}
