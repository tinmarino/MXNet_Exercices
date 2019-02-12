use strict; use warnings; use v5.26;

use LWP::UserAgent ();  # For download
use AI::MXNet ('mx', 'nd');   # For Neuronal Networks
use PDL;             # Perl Data Language for Images "parsing"
use PDL::Math;
use PDL::IO::Image;     # To dump the images
use PDL::IO::FlexRaw;    # To read images


# Placeholder for the input layer
my $data = mx->sym->Variable('data');


sub p {
    if(our $verbose){ say @_ };
}


# Download Mnist data (s_url, b_force -> file)
sub download_data {
    my $url = shift;
    my $force_download = shift;

    p "Loading " . $url;

    my $fname = (split /\//, $url)[-1];

    my $ua = LWP::UserAgent->new();
    if($force_download or not -f $fname) {
        $ua->get($url, ':content_file' => $fname);
    }
    return $fname;
}


# Convert (img -> 4d)
sub to4d {
    my($img) = @_;
    return $img->reshape(28, 28, 1, ($img->dims)[2])->float / 255;
}


# Helper
sub print_pdl{
    my $pdl = shift;
    my $flex = pdl(0.01);
    $pdl = PDL::Math::floor($pdl / $flex) * $flex ;
    p "Score matrix:", $pdl;
}


# Return a nn fully connected <- data placeholder
sub nn_perceptron {
    # Get the image
    my($data) = @_;
    
    # Flatten the image
    $data = mx->sym->Flatten(data => $data);

    # 1/ The first fully-connected layer and the corresponding activation function
    # TODO read the doc about how many activation type I can get (like relu)
    # my $fc1    = mx->sym->FullyConnected(
    #     data => $data,
    #     name => 'fct1',
    #     num_hidden => 128);
    # my $act1 = mx->sym->Activation(
    #     data => $fc1,
    #     name => 'relu1',
    #     act_type => "relu");

    # 2/ The second fully-connected layer and the corresponding activation function
    my $fc2    = mx->sym->FullyConnected(
        data => $data,
        name => 'fct2',
        num_hidden => 64);
    my $act2 = mx->sym->Activation(
        data => $fc2,
        name => 'relu2',
        act_type => "relu");

    # 3/ MNIST has 10 classes
    # It is common to put a softmax at the end and a last vecotr of size of the output (I have 0..9 possibilities)
    my $fc3    = mx->sym->FullyConnected(
        data => $act2,
        name => 'fct3',
        num_hidden => 10);
    # Softmax with cross entropy loss
    my $mlp = mx->sym->SoftmaxOutput(
        data => $fc3,
        name => 'softmax');

    return $mlp;
}


# Return a nn convulutional <- data placeholder
sub nn_conv {
    my($data) = @_;
    # Epoch[9] Batch [200]      Speed: 1625.07 samples/sec      Train-accuracy=0.992090
    # Epoch[9] Batch [400]      Speed: 1630.12 samples/sec      Train-accuracy=0.992850
    # Epoch[9] Train-accuracy=0.991357
    # Epoch[9] Time cost=36.817
    # Epoch[9] Validation-accuracy=0.988100

    my $conv1= mx->symbol->Convolution(data => $data, name => 'conv1', num_filter => 20, kernel => [5,5], stride => [2,2]);
    my $bn1  = mx->symbol->BatchNorm(data => $conv1, name => "bn1");
    my $act1 = mx->symbol->Activation(data => $bn1, name => 'relu1', act_type => "relu");
    my $mp1  = mx->symbol->Pooling(data => $act1, name => 'mp1', kernel => [2,2], stride =>[1,1], pool_type=>'max');

    my $conv2= mx->symbol->Convolution(data => $mp1, name => 'conv2', num_filter => 50, kernel=>[3,3], stride=>[2,2]);
    my $bn2  = mx->symbol->BatchNorm(data => $conv2, name=>"bn2");
    my $act2 = mx->symbol->Activation(data => $bn2, name=>'relu2', act_type=>"relu");
    my $mp2  = mx->symbol->Pooling(data => $act2, name => 'mp2', kernel=>[2,2], stride=>[1,1], pool_type=>'max');


    my $fl   = mx->symbol->Flatten(data => $mp2, name=>"flatten");
    my $fc1  = mx->symbol->FullyConnected(data => $fl,  name=>"fc1", num_hidden=>100);
    my $act3 = mx->symbol->Activation(data => $fc1, name=>'relu3', act_type=>"relu");
    my $fc2  = mx->symbol->FullyConnected(data => $act3, name=>'fc2', num_hidden=>30);
    my $act4 = mx->symbol->Activation(data => $fc2, name=>'relu4', act_type=>"relu");
    my $fc3  = mx->symbol->FullyConnected(data => $act4, name=>'fc3', num_hidden=>10);
    my $softmax = mx->symbol->SoftmaxOutput(data => $fc3, name => 'softmax');
    return $softmax;
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
    my $path = shift;
    unless ($path) {die "read_iamge no argument !!"}

    # 1/ Read HD
    my $im = PDL::IO::Image->new_from_file($path);
    p "Readen im: " . get_image_info $im;

    # 2.1/ Convert image
    $im->convert_image_type("BITMAP"); # Usually useless
    $im->rescale(28, 28);
    my $b_keep_alpha = not $im->get_colors_used;
    unless ($b_keep_alpha){
        $im->color_to_8bpp;
    }
    $im->save("middle_img.png");
    p "Transformed im: " . get_image_info $im;

    # 2.2/ Convert format
    my $pdl = $im->pixels_to_pdl();
    p "Initial pdl " . $pdl->info; 
	$pdl->set_datatype($PDL::Types::PDL_B);
    # Get alpha channel TODO according to image type
    if ($b_keep_alpha){
        $pdl = $pdl->slice(':,:,3');
    }
    # Force reshape for all to fall in line uselessly in 3 dims
    $pdl->reshape(28, 28, 1);
    p "Final pdl " . $pdl->info; 

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


