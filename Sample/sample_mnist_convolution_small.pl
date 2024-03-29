## Convolutional NN for recognizing hand-written digits in MNIST dataset
## It's considered "Hello, World" for Neural Networks
## For more info about the MNIST problem please refer to http://neuralnetworksanddeeplearning.com/chap1.html
use strict; use warnings; use v5.26;
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(GetMNIST_ubyte);
use Test::More tests => 1;


# symbol net
my $batch_size = 100;
### model
my $data = mx->symbol->Variable('data');
my $conv1= mx->symbol->Convolution(data => $data, name => 'conv1', num_filter => 32, kernel => [3,3], stride => [2,2]);
my $bn1  = mx->symbol->BatchNorm(data => $conv1, name => "bn1");
my $act1 = mx->symbol->Activation(data => $bn1, name => 'relu1', act_type => "relu");
my $mp1  = mx->symbol->Pooling(data => $act1, name => 'mp1', kernel => [2,2], stride =>[2,2], pool_type=>'max');
my $conv2= mx->symbol->Convolution(data => $mp1, name => 'conv2', num_filter => 32, kernel=>[3,3], stride=>[2,2]);
my $bn2  = mx->symbol->BatchNorm(data => $conv2, name=>"bn2");
my $act2 = mx->symbol->Activation(data => $bn2, name=>'relu2', act_type=>"relu");
my $mp2  = mx->symbol->Pooling(data => $act2, name => 'mp2', kernel=>[2,2], stride=>[2,2], pool_type=>'max');
my $fl   = mx->symbol->Flatten(data => $mp2, name=>"flatten");
my $fc1  = mx->symbol->FullyConnected(data => $fl,  name=>"fc1", num_hidden=>30);
my $act3 = mx->symbol->Activation(data => $fc1, name=>'relu3', act_type=>"relu");
my $fc2  = mx->symbol->FullyConnected(data => $act3, name=>'fc2', num_hidden=>10);
my $softmax = mx->symbol->SoftmaxOutput(data => $fc2, name => 'softmax');
# check data
GetMNIST_ubyte();
my $train_dataiter = mx->io->MNISTIter({
    image=>"data/train-images-idx3-ubyte",
    label=>"data/train-labels-idx1-ubyte",
    data_shape=>[1, 28, 28],
    batch_size=>$batch_size, shuffle=>1, flat=>0, silent=>0, seed=>10});
my $val_dataiter = mx->io->MNISTIter({
    image=>"data/t10k-images-idx3-ubyte",
    label=>"data/t10k-labels-idx1-ubyte",
    data_shape=>[1, 28, 28],
    batch_size=>$batch_size, shuffle=>1, flat=>0, silent=>0});
my $n_epoch = 1;
my $mod = mx->mod->new(symbol => $softmax);
$mod->fit(
    $train_dataiter,
    eval_data => $val_dataiter,
    optimizer_params=>{learning_rate=>0.01, momentum=> 0.9},
    num_epoch=>$n_epoch
);
my $res = $mod->score($val_dataiter, mx->metric->create('acc'));
ok($res->{accuracy} > 0.8);
