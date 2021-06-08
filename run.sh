#!/bin/bash

nvcc main.cu alltoall.cu
./a.out 5 1
rm a.out
