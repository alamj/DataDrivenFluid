/*
  Copyright (C) Dr. Jahrul Alam (alamj@mun.ca)

  This is free software distributed for teaching purposes.
*/

#include <torch/torch.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cassert>

#include <vector>
#include "xtensor/xarray.hpp"
#include "xtensor/xbuilder.hpp"

#include <libawcm/xplt.h>

// CSV output
#include <fstream>
#include <sstream>

using namespace std;
using namespace torch::indexing;

template <class T>
xt::xarray<T> to_xtensor(torch::Tensor& tensor)
{  
  std::vector<T> data(tensor.data_ptr<T>(), tensor.data_ptr<T>() + tensor.numel() );
  std::size_t N = static_cast<std::size_t>(tensor.sizes()[0]);
  std::vector<std::size_t> shape = {N,1};
    
  return xt::adapt(data, shape); 
}

//c++ xfunc.cxx -I. -I /opt/include/ -I /opt/include/libtorch/include/ -I /opt/include/libtorch/include/torch/csrc/api/include/ -I /opt/include/libtorch/include/ -I /usr/local/include/matplot/ --std=c++20 -Wno-return-type -Wl,-rpath,/opt/include/libtorch/lib /opt/include/libtorch/lib/libtorch.so /opt/include/libtorch/lib/libc10.so /opt/include/libtorch/lib/libtorch_cpu.so -lfftw3 -lcblas -llapack -L /usr/local/lib64/Matplot++/ -lmatplot -lnodesoup -o awcm

int main(int argc, char *argv[])
{

  auto str = torch::kCPU;
  torch::Device device(str);
  torch::manual_seed(123456);
    
  // Construct the nerual network model using `torch::Sequential` 
  torch::nn::Sequential model (
			       torch::nn::Linear(1,256), 
			       torch::nn::Tanh(),
			       torch::nn::Linear(256,64), 
			       torch::nn::Tanh(),
			       torch::nn::Linear(64,32), 
			       torch::nn::Tanh(),
			       torch::nn::Linear(32,1)
			       );
  
  /*
    Prepare sample data by sampling f(x) at N points

    To illustrate coding,
    sample f(x) using xtensor and convert it to torch::Tensor
  */

  static const int N = 2000;
  xt::xarray<float> X = xt::linspace<double>(0, 4*std::numbers::pi,N);
  xt::xarray<float> F = xt::cos(X);
  
  torch::Tensor x = torch::from_blob(X.data(), {N,1}, torch::dtype(torch::kFloat32)).to(device);
  torch::Tensor y = torch::from_blob(F.data(), {N,1}, torch::dtype(torch::kFloat32)).to(device);

  
  //libawcm::xplt::plot(xarr.begin(),xarr.end(), xu.begin());
  //libawcm::xplt::draw();

  // random shuffle
  torch::Tensor shuffled_indices = torch::randperm( N, torch::TensorOptions().dtype(at::kLong)  ); 

  // take only 10% of the data for training
  auto n_val = int (0.15 * N);
  torch::Tensor train_idx = 
    shuffled_indices.index({Slice(0, n_val)}); 
  torch::Tensor x_train = x.index({train_idx});
  torch::Tensor y_train = y.index({train_idx});

  // TO TRAIN THE MODEL, set up optimizer
  torch::optim::Adam optimizer(model->parameters(), 0.01);
  torch::Tensor train_pred = torch::zeros_like(x_train);
  torch::Tensor loss_values = torch::zeros_like(x_train);

  // Run the optimizer for 2000 iterations
  for (size_t epoch = 1; epoch <= 2000; ++epoch) {
    optimizer.zero_grad();
    
    train_pred = model->forward(x_train);
    
    loss_values = torch::mse_loss(train_pred, y_train); 
    loss_values.backward(); 
    
    optimizer.step();
      
    /* Report the error with respect to y_train. */
    double max_loss = loss_values.max().item<double>();
    cout << "Epoch " << epoch 
	 << ", max(loss_values) = " << max_loss << endl;
  }
  

  /*
    predict f(X) for all values of X
    Remark: in this test, X includes those 10% points that
    were also used for training.

    We could create a new X for the domain of f(x), and make predictions.
   */
  torch::Tensor ypred = model->forward(x);

  // convert to xtensor for ploting purpurpose
  xt::xarray<float> y_mod = to_xtensor<float>(ypred);

  xt::xarray<float> X_train = to_xtensor<float>(x_train);
  xt::xarray<float> Y_train = to_xtensor<float>(y_train);

  libawcm::xplt::plot(X_train.begin(), X_train.end(), Y_train.begin(),"o");
  libawcm::xplt::draw();
  
  // create a scatter plot
  libawcm::xplt::plot(F.begin(),F.end(), y_mod.begin(),"k.");
  libawcm::xplt::draw();

  std::cout << "training data size = " << x_train.sizes()[0] << std::endl;

  return 0;
}
