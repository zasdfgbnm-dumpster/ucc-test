#!/bin/bash

export UCX_TLS=tcp
rm -f *.bin a.out *.lock

nvcc utils.cpp exp1.cu -std=c++17 -I $UCC_HOME/include -L $UCC_HOME/lib -lucc -lucs -lucp

./a.out 2 0 &
./a.out 2 1 &

wait
