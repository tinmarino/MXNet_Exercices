use strict; use warnings; use v5.26;

use LWP::UserAgent ();  # For download
use AI::MXNet ('mx', 'nd');   # For Neuronal Networks
use PDL ();             # Perl Data Language for Images "parsing"
use PDL::IO::Image;     # To dump the images
use PDL::IO::FlexRaw;    # To read images

# Add script dir to path
use File::Basename;
use lib dirname (__FILE__);

# Load models
use ia_mnist_model qw/nn_perceptron/;


print "--> Starting Script\n";
my($magic, $num, $rows, $cols);


sub read_data {
  my($label_url, $image_url) = @_;

  open my($flbl), '<:gzip', download_data($label_url, 1);
  read $flbl, my($buf), 8;
  ($magic, $num) = unpack 'N2', $buf;
  my $label = PDL->new();
  $label->set_datatype($PDL::Types::PDL_B);
  $label->setdims([ $num ]);
  read $flbl, ${$label->get_dataref}, $num;
  $label->upd_data();

  open my($fimg), '<:gzip', download_data($image_url, 1);
  read $fimg, $buf, 16;
  ($magic, $num, $rows, $cols) = unpack 'N4', $buf;
  my $image = PDL->new();
  $image->set_datatype($PDL::Types::PDL_B);
  $image->setdims([ $rows, $cols, $num ]);
  read $fimg, ${$image->get_dataref}, $num * $rows * $cols;
  $image->upd_data();

  # Save it
  save_image($image);
  return($label, $image);
}

sub save_image {
  # Many images
  my($image) = @_[0];

  # Write main
  my $im = $image->slice(":,:,:9")->copy;
  $im->set_datatype($PDL::Types::PDL_B);
  $im->setdims([ $rows, $cols, 10]);
  $im->upd_data();
  print "typeof to_dump  for main  : " . ref($im) . " and length " . PDL::nelem($im) . "\n";
  writeflex("testmain.dp", $im);


  # Slice
  for my $i (0..9){
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


# TODO # Fix the seed
# TODO kO  # Set the compute context, GPU is available otherwise CPU
# Create a place holder variable for the input data

################################################################################

# Placeholder for the input layer
my $data = mx->sym->Variable('data');



# Read the data from internet
my $path='http://yann.lecun.com/exdb/mnist/';
print "---> Donwloading training\n";
my($train_lbl, $train_img) = read_data(
    "${path}train-labels-idx1-ubyte.gz", "${path}train-images-idx3-ubyte.gz");
print "---> Donwloading validation\n";
my($val_lbl, $val_img) = read_data(
    "${path}t10k-labels-idx1-ubyte.gz", "${path}t10k-images-idx3-ubyte.gz");



# Define the model
# TODO get gpu => convolutional
#my $mlp = $ARGV[0] ? nn_conv($data) : nn_perceptron($data);
my $mlp = nn_perceptron($data);
my $ctx = $ARGV[0] ? mx->gpu(1) : mx->cpu();
my $model = mx->mod->Module(
  symbol => $mlp,
  context => $ctx,
  );


# Work
sub fit {
  say "--> Starting Fit";

  # Set params
  my $batch_size = 100;
  # Please keep the  declaration of nn that way (with the leading , and newline
  my $train_iter = mx->io->NDArrayIter(
	  data => to4d($train_img),
	  label => $train_lbl,
	  batch_size => $batch_size,
	  shuffle => 1,
  );
  my $val_iter = mx->io->NDArrayIter(
	  data => to4d($val_img),
	  label => $val_lbl,
	  batch_size => $batch_size,
  );

  $model->fit(
    $train_iter,  # train data
    num_epoch => 8,  # train for at most 10 dataset passes
    eval_data => $val_iter,  # validation data
    optimizer => 'adam',  # use SGD to train  tochastic gradient descent
    # optimizer_params => {'learning_rate' => 0.1},  # use fixed learning rate
    eval_metric => 'acc',  # report accuracy during training
    batch_end_callback => mx->callback->Speedometer($batch_size, 200), # output progress for each 100 data batches
  );
}



fit;

$model->save_checkpoint('mycheckpoint.dp', 3);




print "<-- Script Finished\n";
