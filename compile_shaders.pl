#!/usr/bin/perl
use strict;
use warnings;

use File::Find;

find(\&wanted, ".");

sub wanted {
        return unless -f;
        return unless /\.glsl$/;
        my $src = $_;
        (my $dst = $src) =~ s/\.glsl/.spv/;
	my $stage;
	if ($_ =~ /\.vs\./) {
	    $stage = "vertex";
	} elsif ($_ =~ /\.fs\./) {
	    $stage = "fragent";
	} else {
	    $stage = "compute";
	}
	my $cmd = "glslc -fshader-stage=$stage $src -o $dst";
	print "$cmd\n";
	system $cmd;
}
