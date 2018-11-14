#!/usr/bin/perl

# TODO load the pdl and the model and find good digit
use strict; use warnings; use v5.26;

use LWP::UserAgent ();  # For download
use AI::MXNet ('mx', 'nd');   # For Neuronal Networks
use PDL ();             # Perl Data Language for Images "parsing"
use PDL::IO::Image;     # To dump the images
use PDL::IO::FlexRaw;    # To read images

# Add script dir to path
use File::Basename;
use lib dirname (__FILE__);

# Include models
use ia_mnist_model qw/nn_perceptron/;

say "--> Starting Script V1.0";


# Load PDL image
sub read_image {
	my $im = readflex('testmain.dp');
	#open my($test1), "test1.dp";
	#my $im = PDL->new();
	my $rows = 28;
	my $cols = 28;
	my $num = 10;
	$im->set_datatype($PDL::Types::PDL_B);
	$im->setdims([ $rows, $cols, $num]);
	print "typeof to_dump " . ref($im) . " and length " . PDL::nelem($im) . "\n";
	$im = to4d($im);
	return $im;
}

my $im = read_image;

open my($flbl), '<:gzip', download_data(1);
my $label = PDL->new();
$label->set_datatype($PDL::Types::PDL_B);
$label->setdims([ 10 ]);
read $flbl, ${$label->get_dataref}, 10;
$label->upd_data();

my $test_iter = mx->io->NDArrayIter(
    data => $im,
    label => $label,
);




# Load IA model
my $data = mx->sym->Variable('data');
my $mlp = nn_perceptron($data);
my $model = mx->mod->Module(symbol=> $mlp,);
# allocate memory given the input data and label shapes
$model->bind(
  data_shapes => $test_iter->provide_data, 
  label_shapes => $test_iter->provide_label,
);
my ($sym, $arg_params, $aux_params ) = $model->load_checkpoint('mycheckpoint.dp', 3);
$model->set_params($arg_params, $aux_params);

# Score with the accuracy metric
my $val = $model->predict($test_iter);
say "score is ", $val->aspdl;





say "--> End Script\n";
