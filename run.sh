#!/bin/bash

nvcc main.cu alltoall.cu -I $UCC_HOME/include
./a.out 5 1
rm a.out
