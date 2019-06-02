use strict; use warnings; use v5.26;

#use PDL::Ufunc;              # for max index in array

# Add script dir to path
use File::Basename;
use lib dirname (__FILE__);

# Include models
use mnist_model qw/nn_perceptron/;

# Usage
sub usage{
    "Usage: " . __FILE__ . " img1.jpg [img2.jpg [...]]";
}

# Main 
sub main {
    # Hi
    our $verbose = 1;
    p "--> Starting " . __FILE__ . " V1.0";
    $ARGV[0] || die usage;

    for my $path (@ARGV){
        # Load
        my $test_iter = mx->io->NDArrayIter(
            data => read_image $path);
        my $model = read_model $test_iter;

        # Perdict -> Matrix probability
        my $val = $model->predict($test_iter);
        my $pdl = $val->aspdl;
        print_pdl $pdl;

        # Get max <- Matrix
        my $i_res = $pdl->maximum_ind->index(0);
        my $out = sprintf "%-30s is a    %d", $path, $i_res;
        say $out;
    }

    # Bye
    p "<-- End Script\n";
}

# Work
main;
