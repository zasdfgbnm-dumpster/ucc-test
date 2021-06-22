#!/bin/bash
set -eux

rm -rf *.bin a.out *.lock

export UCX_TLS=tcp
export UCX_WARN_UNUSED_ENV_VARS=n

# cpu

g++ utils.cpp main.cpp alltoall.cpp -std=c++17 -I $UCC_HOME/include -L $UCC_HOME/lib -lucc -lucs -lucp

./a.out 2 0 &
./a.out 2 1 &

wait
# rm a.out
exit

# cuda

nvcc utils.cpp main.cu alltoall.cu -std=c++17 -I $UCC_HOME/include -L $UCC_HOME/lib -lucc -lucs -lucp

# nsys nvprof ./a.out 2 0 &
# nsys nvprof ./a.out 2 1 &

./a.out 2 0 &
./a.out 2 1 &

# rm a.out

wait