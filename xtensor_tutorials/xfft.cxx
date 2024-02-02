/*
  Author: Dr. Jahrul Alam (alamj@mun.ca)
  This code is a part of AWCM++ (Advanced Weighted Residual Collocation Methodology in C++)
*/

#include <iostream>
#include <fstream>
#include <bit>
#include <filesystem> // C++17
#include <chrono>
#include <charconv>

#include <vector>
#include <map>
#include <set>
#include <numbers>

#include <boost/algorithm/string.hpp>
#include <boost/math/statistics/bivariate_statistics.hpp>

#include "xutils.h"

/*
  /opt/include: where xtensor and other libraries are installed
  -lfftw3: to use fftw through xtensor-fftw
  -lcblas -llapack: to use linear algebra through xtensor-blas
  /usr/local/lib64/Matplot++/: where matplot++ library lives
  -lmatplot -lnodesoup: to use matplot++

  compiles as:
  c++ xfft.cxx -I /opt/include/ -I /usr/local/include/matplot/ --std=c++20 -lfftw3 -lcblas -llapack -L /usr/local/lib64/Matplot++/ -lmatplot -lnodesoup -o awcm  
*/

  
int main()
{

  /*
    Sample a function f(x) at two given wave numbers (50,600)
    The number of samples n and the length of the domain L
    are two variables for testing how FFT can capture the wavenumbers.
  */
  int n = 2048;
  double L = 4.0;
  double dx = L/n;
  
  xt::xarray<double> x = xt::linspace<double>(0,L,n);
  xdouble u = sin(2*pi*50*x/L) + sin(2*pi*600*x/L);

  //libawcm::xplt::plot(x.begin(),x.end(), u.begin());
  //libawcm::xplt::draw();

  /*
    Due to conjugate symmetry
    the size of uHat is n/2+1
    where the first element uHat[0]
    corresponds to k=0
  */
  
  auto uHat = xt::fftw::rfft(u);
  xdouble psd = xt::real(uHat*xt::conj(uHat));

  /*
    Calculate energy in Fourier space
    Take the odd and the rest of the spectrum
    To account for conjugate symmetry,
    double the rest.
    Take average over n numbers.

    Using conjugate symmetry, size of psd is n/2+1
    Wavenumber range is k = [0, n/2]
  */

  auto k = xt::arange<double>(0,psd.shape()[0],1);

  /*
    Verify how accurately the energy is prserved
  */
  auto odd = xt::view(psd,xt::range(0,1));
  auto rst = xt::view(psd,xt::range(1,_));

  auto energy = (xt::sum(xt::abs(odd)) + 2*xt::sum(xt::abs(rst)))/n;
  std::cout << "Energy of Fourier modes: " << energy << ", Actual energy: " << xt::sum(xt::pow(u,2)) << std::endl;

  //libawcm::xplt::plot(k.begin(),k.end(), xt::abs(psd).begin());
  //libawcm::xplt::draw();

  return 0;
}

