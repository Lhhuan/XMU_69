#!/usr/bin/perl
use warnings;
use strict; 
use utf8;
use File::Basename;
use List::Util qw/max min/;
use Env qw(PATH);
use Parallel::ForkManager;


my @cj =(5..9);
my $pm = Parallel::ForkManager->new(5); 
# my $pid = $pm->start and next; #开始多线程
foreach my $j(@cj){
    my $pid = $pm->start and next; #开始多线程
    print "$j\n"; 
    $pm->finish;  #多线程结束  
}
