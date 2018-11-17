# TODO load the pdl and the model and find good digit
use strict; use warnings; use v5.26;

use LWP::UserAgent ();  # For download
use AI::MXNet ('mx', 'nd');   # For Neuronal Networks
use PDL;             # Perl Data Language for Images "parsing"
use PDL::Math;
use PDL::IO::Image;     # To dump the images
use PDL::IO::FlexRaw;    # To read images

# Add script dir to path
use File::Basename;
use lib dirname (__FILE__);

# Include models
use ia_mnist_model qw/nn_perceptron/;



sub usage{
    "Usage: " . __FILE__ . " image_to_read.jpg";
}


# Read image on HD and convert it to 1 bytes x 28x28
sub read_image{
    my $path = $ARGV[0];
    unless ($path) {die "Error: " . usage}
    my $pdl1 = rimage($path);
    say $pdl1->info; 
    my $im = to4d($pdl1);
    return $im;
}


my $test_iter = mx->io->NDArrayIter(
    data => read_image,
);

# Load IA model
sub read_model{
	my $data = mx->sym->Variable('data');
	my $mlp = nn_perceptron($data);
	my $model = mx->mod->Module(symbol=> $mlp,);
	my ($sym, $arg_params, $aux_params ) = $model->load_checkpoint('mycheckpoint.dp', 3);
	$model->bind(
		data_shapes => $test_iter->provide_data,
	);
	$model->set_params($arg_params, $aux_params);
	return $model;
}


sub main {
    # Hi
    say "--> Starting " . __FILE__ . " V1.0";

    # Load
    my $image = read_image;
    my $model = read_model $image;

    # Perdict
    my $val = $model->predict($test_iter);

    # Print
    print_pdl $val->aspdl;

    # Bye
    say "--> End Script\n";
}

main;
