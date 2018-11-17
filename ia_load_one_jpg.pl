# TODO load the pdl and the model and find good digit
use strict; use warnings; use v5.26;

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

# Strbuild info <- PDL::IO::Image
sub get_image_info{
    my $pimage1 = shift;
    use overload '.' => sub { shift . "\n" . shift  };
    return "\n"
        . 'width       = ' . $pimage1->get_width . "\n"
        . 'height      = ' . $pimage1->get_height . "\n"
        . 'image_type  = ' . $pimage1->get_image_type . "\n"
        . 'color_type  = ' . $pimage1->get_color_type . "\n"
        . 'colors_used = ' . $pimage1->get_colors_used . "\n"
        . 'bpp         = ' . $pimage1->get_bpp . "\n";
}


# Read image on HD and convert it to 1 bytes x 28x28
sub read_image{
    # 0/ In
    my $path = $ARGV[0];
    unless ($path) {die "Error: " . usage}

    # 1/ Read HD
    my $im = PDL::IO::Image->new_from_file($path);
    say "Readen im: " . get_image_info $im;

    # 2.1/ Convert image
    $im->convert_image_type("BITMAP"); # Usually useless
    $im->rescale(28, 28);
    my $b_keep_alpha = not $im->get_colors_used;
    unless ($b_keep_alpha){
        $im->color_to_8bpp;
    }
    $im->save("middle_img.png");
    say "Transformed im: " . get_image_info $im;

    # 2.2/ Convert format
    my $pdl = $im->pixels_to_pdl();
    say "Initial pdl " . $pdl->info; 
	$pdl->set_datatype($PDL::Types::PDL_B);
    # Get alpha channel TODO according to image type
    if ($b_keep_alpha){
        $pdl = $pdl->slice(':,:,3');
    }
    # Force reshape for all to fall in line uselessly in 3 dims
    $pdl->reshape(28, 28, 1)->float / 255;
    say "Final pdl " . $pdl->info; 

    # 3/ Ret
    return $pdl;
}


# Load IA model
sub read_model{
    my $test_iter = shift;
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
    my $test_iter = mx->io->NDArrayIter(
        data => read_image);
    my $model = read_model $test_iter;

    # Perdict -> Matrix probability
    my $val = $model->predict($test_iter);
    my $pdl = $val->aspdl;
    print_pdl $pdl;

    # Get max <- Matrix
    my $i_res = $pdl->maximum_ind->slice(0); 
    say "An we got a", $i_res;

    # Bye
    say "--> End Script\n";
}

main;
