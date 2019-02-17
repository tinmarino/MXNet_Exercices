#!/usr/bin/perl
use strict;
use warnings;
use v5.26;
# derived from http://mxnet.io/tutorials/python/mnist.html
use LWP::UserAgent ();
use PDL ();
use AI::MXNet ('mx');


sub download_data {
    my($url, $force_download) = @_;
    my $fname = (split /\//, $url)[-1];
    if (@_ > 1) {
      say "I am not downloading";
      return $fname;
    }
    my $ua = LWP::UserAgent->new();
    $force_download = 1 if @_ < 2;
    if($force_download or not -f $fname) {
        $ua->get($url, ':content_file' => $fname);
    }
    return $fname;
}

sub read_data {
    my($label_url, $image_url) = @_;
    my($magic, $num, $rows, $cols);

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
    return($label, $image);
}

my $path='http://yann.lecun.com/exdb/mnist/';
my($train_lbl, $train_img) = read_data(
    "${path}train-labels-idx1-ubyte.gz", "${path}train-images-idx3-ubyte.gz");
my($val_lbl, $val_img) = read_data(
    "${path}t10k-labels-idx1-ubyte.gz", "${path}t10k-images-idx3-ubyte.gz");

sub to4d {
    my($img) = @_;
    return $img->reshape(28, 28, 1, ($img->dims)[2])->float / 255;
}

my $batch_size = 100;
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

# Create a place holder variable for the input data
my $data = mx->sym->Variable('data');

sub nn_fc {
    # Epoch[9] Train-accuracy=0.978889
    # Epoch[9] Time cost=145.437
    # Epoch[9] Validation-accuracy=0.964600
    my($data) = @_;

    # Flatten the data from 4-D shape (batch_size, num_channel, width, height) 
    # into 2-D (batch_size, num_channel*width*height)
    $data = mx->sym->Flatten(data => $data);

    # The first fully-connected layer
#    my $fc1  = mx->sym->FullyConnected(data => $data, name => 'fc1', num_hidden => 128);
#    # Apply relu to the output of the first fully-connnected layer
#    my $act1 = mx->sym->Activation(data => $fc1, name => 'relu1', act_type => "relu");

    # The second fully-connected layer and the according activation function
    my $fc2  = mx->sym->FullyConnected(data => $data, name => 'fc2', num_hidden => 64);
    my $act2 = mx->sym->Activation(data => $fc2, name => 'relu2', act_type => "relu");

    # The thrid fully-connected layer, note that the hidden size should be 10, which is the number of unique digits
    my $fc3  = mx->sym->FullyConnected(data => $act2, name => 'fc3', num_hidden => 10);
    # The softmax and loss layer
    my $mlp  = mx->sym->SoftmaxOutput(data => $fc3, name => 'softmax');
    say "Yes Your are in fc";
    return $mlp;
}

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

my $mlp = $ARGV[0] ? nn_conv($data) : nn_fc($data);
#We visualize the network structure with output size (the batch_size is ignored.)
#my $shape = { data => [ $batch_size, 1, 28, 28 ] };
#show_network(mx->viz->plot_network($mlp, shape => $shape));

my $model = mx->mod->Module(
    symbol => $mlp,       # network structure
);
$model->fit(
    $train_iter,       # training data
    num_epoch => 10,      # number of data passes for training 
    eval_data => $val_iter, # validation data
    batch_end_callback => mx->callback->Speedometer($batch_size, 200), # output progress for each 200 data batches
    optimizer => 'adam',
);
