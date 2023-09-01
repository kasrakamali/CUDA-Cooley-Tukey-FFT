# CUDA Cooley-Tukey FFT

## Overview

This repository contains CUDA code for computing the Fast Fourier Transform (FFT) using the Cooley-Tukey algorithm with mixed radix (radix 2 and 4). The implementation is optimized for NVIDIA GPUs.

## Background

The one-dimensional N-point DFT formula is given by:

```
X[k] = ∑ x[n] * exp(-j * 2π * n * k / N), k = 0, 1, ..., N-1
```

The algorithm mentioned has a time complexity of O(N^2). However, assuming N=2^M, we can use a divide-and-conquer approach to calculate X[k] by splitting x[n] into two parts - odd values of n and even values of n. This approach is called the Cooley-Tukey algorithm and has a time complexity of O(N logN). The algorithm has logN stages, and each stage involves N/2 butterfly operations. Each butterfly operation takes 2 complex values as input and produces 2 complex values as output.

The Cooley-Tukey algorithm can be used to rewrite the equation in a different way:

```
X[k] = X1[k] + W_N^k * X2[k], k = 0, 1, ..., N/2-1
```

where `W_N^k = exp(-j * 2π * k / N)`, `X1[k]` is the N/2 point DFT of the even half of the signal `x[n]`, and `X2[k]` is the N/2 point DFT of the odd half of the signal `x[n]`.

## Usage

The program prompts the user to enter the size of the input array. After entering the size, the program generates a random input signal and computes its FFT using the Cooley-Tukey algorithm with mixed radix.

To compile the code, run the following command:

```
nvcc -O2 fft_main.cu fft.cu -o fft
```

To execute the code, run the following command:

```
./fft M
```

where M is the log2 of the number of points in the DFT.

## Performance

The performance of the FFT implementation depends on the size of the input array and the GPU architecture. For large arrays and newer GPUs, the implementation should be able to achieve near-peak performance.

## References

- [NTU DSP Course: FFT](http://www.cmlab.csie.ntu.edu.tw/cml/dsp/training/coding/transform/fft.html)
- [Wikipedia: Fast Fourier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform)
- [Wikipedia: Cooley-Tukey FFT algorithm](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)
