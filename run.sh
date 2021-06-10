#!/bin/bash

nvcc main.cu alltoall.cu -I $UCC_HOME/include -L $UCC_HOME/lib -lucc -lucs -lucp
UCX_TLS=tcp ./a.out 2 1
# rm a.out
