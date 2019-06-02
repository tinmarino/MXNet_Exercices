use strict; use warnings; use v5.26;

use LWP::UserAgent ();          # For download
use AI::MXNet ('mx', 'nd');     # For Neuronal Networks
use PDL;                        # Perl Data Language for Images "parsing"
use PDL::Math;
use PDL::IO::Image;             # To dump the images
use PDL::IO::FlexRaw;           # To read images
use Function::Parameters;       # Define function with parameters inside

# Add script dir to path
use File::Basename;
use lib dirname (__FILE__);

# Include models
use mnist_model qw/nn_perceptron/;



# Load PDL image
sub read_image_local {
	my $im = readflex('data/testmain.dp');
	my $rows = 28;
	my $cols = 28;
	my $num = 10;
	$im->set_datatype($PDL::Types::PDL_B);
	$im->setdims([ $rows, $cols, $num]);
	p "typeof to_dump " . ref($im) . " and length " . PDL::nelem($im) . "\n";
	$im = to4d($im);
    p "Final pdl " . $im->info; 
	return $im;
}


# Load PDL label
sub read_label_local {
    # Declare res
	my $label = PDL->new();
	$label->set_datatype($PDL::Types::PDL_B);
	$label->setdims([ 10 ]);

    # Open
	open my($flbl), '<:gzip', download_data(1);

    # Read and fill
	read $flbl, ${$label->get_dataref}, 10;
	$label->upd_data();

    p "Label :", $label->info;

    # Ret
	return $label;
}


# Load IA model
fun read_model_local($test_iter){
	my $mlp = nn_perceptron;
	my $model = mx->mod->Module(symbol=> $mlp,);
	# allocate memory given the input data and label shapes
	$model->bind(
		data_shapes => $test_iter->provide_data, 
		label_shapes => $test_iter->provide_label,
	);
	my ($sym, $arg_params, $aux_params ) = $model->load_checkpoint(get_checkpoint_filename, 3);
	$model->set_params($arg_params, $aux_params);
	return $model;
}


sub main {
    # Hi
    our $verbose = 1;
    p "--> Starting Script V1.0";

    # Score with the accuracy metric
    my $test_iter = mx->io->NDArrayIter(
        data => read_image_local,
        label => read_label_local,
    );
    my $model = read_model_local($test_iter);

    # Work
    my $val = $model->predict($test_iter);

    # Out
    my $pdl = $val->aspdl;
    my $flex = pdl(0.01);
    $pdl = PDL::Math::floor($pdl / $flex) * $flex ;
    say "score is ", $pdl;

    # Bye
    p "--> End Script\n";
}


main;


