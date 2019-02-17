use strict; use warnings; use v5.26;

use LWP::UserAgent ();  # For download
use AI::MXNet ('mx', 'nd');   # For Neuronal Networks
use PDL ();             # Perl Data Language for Images "parsing"
use PDL::IO::Image;     # To dump the images
use PDL::IO::FlexRaw;    # To read images
use Function::Parameters;

# Add script dir to path
use File::Basename;
use lib dirname (__FILE__);

# Load models
use mnist_model qw/nn_perceptron download_data/;

# TODO hide that
my($magic, $num, $rows, $cols);

# Define the model
sub get_model {
    my $mlp; my $ctx;
    # Placeholder for the input layer
    my $data = mx->sym->Variable('data');
    if ($ARGV[0]) {
        $ctx = mx->gpu();
        $mlp = nn_conv($data);
        say "I am using gpu 0";
    } else {
        $ctx = mx->cpu();
        $mlp = nn_perceptron($data);
        say "I am using (only) cpu";
    }
    my $model = mx->mod->Module(
      symbol => $mlp,
      context => $ctx,
      );
    return $model;
}


################################################################################

sub read_data {
    my($label_url, $image_url) = @_;

    open my($flbl), '<:gzip', download_data($label_url)  or die "Cannot open input labels";
    read $flbl, my($buf), 8;
    ($magic, $num) = unpack 'N2', $buf;
    my $label = PDL->new();
    $label->set_datatype($PDL::Types::PDL_B);
    $label->setdims([ $num ]);
    read $flbl, ${$label->get_dataref}, $num;
    $label->upd_data();

    open my($fimg), '<:gzip', download_data($image_url) or die "Cannot open input images";
    read $fimg, $buf, 16;
    ($magic, $num, $rows, $cols) = unpack 'N4', $buf;
    my $image = PDL->new();
    $image->set_datatype($PDL::Types::PDL_B);
    $image->setdims([ $rows, $cols, $num ]);
    read $fimg, ${$image->get_dataref}, $num * $rows * $cols;
    $image->upd_data();

    # Save it
    #save_image($image);
    return($label, $image);
}


sub save_image {
    # Many images
    my($image) = $_[0];

    # Write main
    my $im = $image->slice(":,:,:9")->copy;
    $im->set_datatype($PDL::Types::PDL_B);
    $im->setdims([ $rows, $cols, 10]);
    $im->upd_data();
    print "typeof to_dump  for main  : " . ref($im) . " and length " . PDL::nelem($im) . "\n";
    writeflex("testmain.dp", $im);


    # Slice
    for my $i (0..9){
        # Set data format
        my $im = $image->slice(":,:,$i")->copy;
        $im->set_datatype($PDL::Types::PDL_B);
        $im->setdims([ $rows, $cols]);
        $im->upd_data();
        print "typeof to_dump  for ${i} : " . ref($im) . " and length " . PDL::nelem($im) . "\n";

        # Write array
        writeflex("test_${i}.dp", $im);

        # Write png
        my $v2 = $im->byte;
        my $pimage2 = PDL::IO::Image->new_from_pdl($v2);
        $pimage2->save("output_${i}.jpg");
    }
}


sub get_array_iter {
    my($lbl, $img) = read_data(shift, shift);
    my $iter = mx->io->NDArrayIter(
	  data => to4d($img),
	  label => $lbl,
	  batch_size => 100,
	  shuffle => 1,
    );
    return $iter;
}


# Work <- Model 
fun fit($model) {
    # Read the data from internet
    my $path='http://yann.lecun.com/exdb/mnist/';
    say "---> Donwloading training";
    my $train_iter = get_array_iter(
        "${path}train-labels-idx1-ubyte.gz", "${path}train-images-idx3-ubyte.gz");
    say "---> Donwloading validation";
    my $val_iter = get_array_iter(
        "${path}t10k-labels-idx1-ubyte.gz", "${path}t10k-images-idx3-ubyte.gz");

    say "--> Starting Fit";
    $model->fit(
        $train_iter,            # train data
        num_epoch => 3,         # train for at most 10 dataset passes
        eval_data => $val_iter, # validation data
        optimizer => 'adam',    # use SGD to train  tochastic gradient descent
        # optimizer_params => {'learning_rate' => 0.1},  # use fixed learning rate
        eval_metric => 'acc',  # report accuracy during training
        batch_end_callback => mx->callback->Speedometer(100, 200), # output progress for each 100 data batches done for batch sie = 100
    );
    return 1;
}


# Remove and save : model
fun save ($model){
    say "Saving model to mycheckpoint.dp";
    unlink 'mycheckpoint.dp';
    $model->save_checkpoint('mycheckpoint.dp', 3);
    return 1;
}


sub main {
    print "--> Starting Script\n";
    my $model = get_model();
    fit($model) and save($model);
    print "<-- Script Finished\n";
}

main;
