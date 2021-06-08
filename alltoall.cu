#include <string>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <random>

constexpr int N = 5;
using T = int;
int world_size;
int rank;

void check(bool condition, std::string msg = "") {
  if (!condition) {
    std::cerr << msg << std::endl;
    exit(1);
  }
}

void check_cuda(cudaError_t err) {
  check(err == cudaSuccess, cudaGetErrorString(err));
}

void check_cuda() {
  check_cuda(cudaGetLastError());
}

std::vector<T *> buffers;

void allocate_buffers() {
  for (int i = 0; i < world_size; i++) {
    void *ptr;
    check_cuda(cudaMallocManaged(&ptr, sizeof(T) * N));
    buffers.push_back(reinterpret_cast<T *>(ptr));
  }
}

void initialize_buffers() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> d(0, N * world_size);

  for (int i = 0; i < world_size; i++) {
    for (int j = 0; j < N; j++) {
      T value = (i == rank ? d(gen): 0);
      buffers[i][j] = value;
    }
  }
}

void print_buffers() {
  for (int i = 0; i < world_size; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << buffers[i][j] << ", ";
    }
    std::cout << std::endl;
  }
}

void alltoall() {}

int main(int argc, char *argv[]) {
  check(argc == 3, "Bad argument");
  world_size = std::stoi(argv[1]);
  rank = std::stoi(argv[2]);
  std::cout << "World size: " << world_size << ", " << "Rank: " << rank << std::endl;

  allocate_buffers();
  initialize_buffers();
  std::cout << std::endl << "Buffers initialized as:" << std::endl;
  print_buffers();
  alltoall();
  std::cout << std::endl << "After alltoall, buffers are:" << std::endl;
  print_buffers();
}