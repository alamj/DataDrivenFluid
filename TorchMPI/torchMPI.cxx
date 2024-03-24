#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#include <torch/torch.h>
#include <iostream>

int main()
{

  auto pg = c10d::ProcessGroupMPI::createProcessGroupMPI();

  // Retrieving MPI environment variables                                                                                                                                                         
  auto numranks = pg->getSize();
  auto rank = pg->getRank();

    
std::cout << numranks << ", " << rank << std::endl;
torch::Device device(torch::kCPU);
  if (torch::cuda::is_available())
    {
      std::cout << "CUDA is available! Using GPU." << std::endl;
      device = torch::Device(torch::kCUDA);
    }
  
torch::Tensor tensor = torch::rand({2, 3}).to(device);
std::cout << tensor << std::endl;
