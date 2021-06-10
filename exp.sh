#!/bin/bash

nvcc exp1.cu -I $UCC_HOME/include -L $UCC_HOME/lib -lucc -lucs -lucp
UCX_TLS=cuda ./a.out
