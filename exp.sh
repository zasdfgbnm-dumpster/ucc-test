#!/bin/bash

export UCX_TLS=tcp
nvcc exp1.cu -I $UCC_HOME/include -L $UCC_HOME/lib -lucc -lucs -lucp
./a.out
