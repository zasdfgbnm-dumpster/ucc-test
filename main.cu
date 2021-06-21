#include <string>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <thread>
#include <chrono>

#include "utils.hpp"

int N = 5;
using T = int;
int world_size;
int rank;

void check_cuda(cudaError_t err) {
  check(err == cudaSuccess, cudaGetErrorString(err));
}

void check_cuda() {
  check_cuda(cudaGetLastError());
}

int get_device() {
  return rank;
}

void set_device() {
  check_cuda(cudaSetDevice(get_device()));
}

void set_device(int i) {
  check_cuda(cudaSetDevice(i));
}

cudaStream_t getStreamFromPool() {
  cudaStream_t stream;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  return stream;
}

cudaStream_t getCurrentCUDAStream() {
  static cudaStream_t stream = getStreamFromPool();
  return stream;
}

T *input;
T *output;

void allocate_buffers() {
  check_cuda(cudaMalloc(&input, sizeof(T) * N * world_size));
  check_cuda(cudaMalloc(&output, sizeof(T) * N * world_size));
}

__global__ void write_value(T *ptr, T value) {
  *ptr = value;
}

void initialize_input() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> d(0, N * world_size);

  for (int i = 0; i < world_size; i++) {
    for (int j = 0; j < N; j++) {
      T value = d(gen);
      write_value<<<1, 1, 0, getCurrentCUDAStream()>>>(input + N * i + j, value);
    }
  }
}

template<typename T>
void print_buffer(T *ptr) {
  cudaStreamSynchronize(getCurrentCUDAStream());
  for (int i = 0; i < world_size; i++) {
    T *host = new T[N];
    check_cuda(cudaMemcpyAsync(host, ptr + N * i, sizeof(T) * N, cudaMemcpyDefault, getCurrentCUDAStream()));
    cudaStreamSynchronize(getCurrentCUDAStream());
    for (int j = 0; j < N; j++) {
      std::cout << host[j] << ", ";
    }
    delete [] host;
    std::cout << std::endl;
  }
}

void alltoall();

int main(int argc, char *argv[]) {
  check(argc == 3, "Bad argument");
  world_size = std::stoi(argv[1]);
  rank = std::stoi(argv[2]);
  std::cout << "World size: " << world_size << ", " << "Rank: " << rank << std::endl;

  set_device();

  allocate_buffers();
  initialize_input();
  std::cout << std::endl << "Buffers initialized as:" << std::endl;
  print_buffer(input);

  alltoall();
  cudaDeviceSynchronize();

  std::this_thread::sleep_for(std::chrono::seconds(rank));
  std::cout << std::endl << "After alltoall, buffers are:" << std::endl;
  print_buffer(output);
}