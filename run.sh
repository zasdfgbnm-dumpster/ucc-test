#!/bin/bash

nvcc main.cu alltoall.cu -I $UCC_HOME/include -L $UCC_HOME/lib -lucc -lucs -lucp
./a.out 5 1
rm a.out
