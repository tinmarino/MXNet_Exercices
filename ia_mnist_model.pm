use strict; use warnings; use v5.26;

say "< Including mnist model module v1.0";

# Placeholder for the input layer
my $data = mx->sym->Variable('data');


sub download_data {
  my $force_download = shift;

  my $url='http://yann.lecun.com/exdb/mnist/';
  my $label_url = "${url}t10k-labels-idx1-ubyte.gz";
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



sub to4d {
    my($img) = @_;
    return $img->reshape(28, 28, 1, ($img->dims)[2])->float / 255;
}


sub print_pdl{
    my $pdl = shift;
    my $flex = pdl(0.01);
    my $pdl = PDL::Math::floor($pdl / $flex) * $flex ;
    say "score is ", $pdl;
}


# Return a nn fully connected <- data placeholder
sub nn_perceptron {
  # Get the image
  my($data) = @_;
  
  # Flatten the image
  $data = mx->sym->Flatten(data => $data);

  # 1/ The first fully-connected layer and the corresponding activation function
  # TODO read the doc about how many activation type I can get (like relu)
  # my $fc1  = mx->sym->FullyConnected(
  #   data => $data,
  #   name => 'fct1',
  #   num_hidden => 128);
  # my $act1 = mx->sym->Activation(
  #   data => $fc1,
  #   name => 'relu1',
  #   act_type => "relu");

  # 2/ The second fully-connected layer and the corresponding activation function
  my $fc2  = mx->sym->FullyConnected(
    data => $data,
    name => 'fct2',
    num_hidden => 64);
  my $act2 = mx->sym->Activation(
    data => $fc2,
    name => 'relu2',
    act_type => "relu");

  # 3/ MNIST has 10 classes
  # It is common to put a softmax at the end and a last vecotr of size of the output (I have 0..9 possibilities)
  my $fc3  = mx->sym->FullyConnected(
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


