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
	my $stage = $_ =~ /\.vs\./ ? "vertex" : "fragment";
	my $cmd = "glslc -fshader-stage=$stage $src -o $dst";
	print "$cmd\n";
	system $cmd;
}
