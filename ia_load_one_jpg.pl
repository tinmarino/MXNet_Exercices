# TODO load the pdl and the model and find good digit
use strict; use warnings; use v5.26;

use Log::Message::Simple qw/debug/;

use LWP::UserAgent ();  # For download
use AI::MXNet ('mx', 'nd');   # For Neuronal Networks
use PDL;             # Perl Data Language for Images "parsing"
use PDL::Math;
use PDL::IO::Image;     # To dump the images
use PDL::IO::FlexRaw;    # To read images
use PDL::Ufunc;              # for max index in array

# Add script dir to path
use File::Basename;
use lib dirname (__FILE__);

# Include models
use ia_mnist_model qw/nn_perceptron/;

sub usage{
    "Usage: " . __FILE__ . " image_to_read.jpg";
}


sub main {
    # Hi
    say "--> Starting " . __FILE__ . " V1.0";
    my $path = $ARGV[0] || die usage;


    # Load
    my $test_iter = mx->io->NDArrayIter(
        data => read_image $path);
    my $model = read_model $test_iter;

    # Perdict -> Matrix probability
    my $val = $model->predict($test_iter);
    my $pdl = $val->aspdl;
    print_pdl $pdl;

    # Get max <- Matrix
    my $i_res = $pdl->maximum_ind->slice(0); 
    say "An we got a", $i_res;

    # Bye
    say "<-- End Script\n";
}

main;
