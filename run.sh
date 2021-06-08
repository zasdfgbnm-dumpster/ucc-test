#!/bin/bash

nvcc alltoall.cu
./a.out 5 1
rm a.out
