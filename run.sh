#!/bin/bash

rm -rf *.bin a.out

export UCX_TLS=tcp

nvcc utils.cpp main.cu alltoall.cu -std=c++17 -I $UCC_HOME/include -L $UCC_HOME/lib -lucc -lucs -lucp

./a.out 2 0 &
./a.out 2 1 &

wait
# rm a.out
