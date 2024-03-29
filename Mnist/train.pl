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
use model qw/download_data/;

my($rows, $cols);  # 28, 28

################################################################################

sub read_data {
    my($label_url, $image_url) = @_;

    my ($magic, $num);

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
    save_image($image);
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
    writeflex("data/testmain.dp", $im);


    # Slice
    for my $i (0..9){
        # Set data format
        my $im = $image->slice(":,:,$i")->copy;
        $im->set_datatype($PDL::Types::PDL_B);
        $im->setdims([ $rows, $cols]);
        $im->upd_data();
        print "typeof to_dump  for ${i} : " . ref($im) . " and length " . PDL::nelem($im) . "\n";

        # Write array
        writeflex("data/test_${i}.dp", $im);

        # Write png
        my $v2 = $im->byte;
        my $pimage2 = PDL::IO::Image->new_from_pdl($v2);
        $pimage2->save("data/output_${i}.jpg");
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
    # Hi
    our $verbose = 1;
    p '--> Going to train on Mnist net hadwritten dataset';

    # Read the data from internet
    my $path='http://yann.lecun.com/exdb/mnist/';
    p "---> Downloading training";
    my $train_iter = get_array_iter(
        "${path}train-labels-idx1-ubyte.gz", "${path}train-images-idx3-ubyte.gz");
    p "---> Downloading validation";
    my $val_iter = get_array_iter(
        "${path}t10k-labels-idx1-ubyte.gz", "${path}t10k-images-idx3-ubyte.gz");

    # Fit
    p "--> Starting Fit";
    $model->fit(
        $train_iter,            # train data
        num_epoch => 3,         # train for at most 10 dataset passes
        eval_data => $val_iter, # validation data
        optimizer => 'adam',    # use SGD to train  tochastic gradient descent
        # optimizer_params => {'learning_rate' => 0.1},  # use fixed learning rate
        eval_metric => 'acc',  # report accuracy during training
        batch_end_callback => mx->callback->Speedometer(100, 200), # output progress for each 100 data batches done for batch sie = 100
    );

    # Ret
    return 1;
}


sub main {
    print "--> Starting Script\n";
    my $model = get_model;
    fit($model) and save_model $model;
    print "<-- Script Finished\n";
}

main;
