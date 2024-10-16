#!/bin/bash

pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

pip install torchdata==0.4.1

pip install torch_geometric==2.1.0.post1

pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl

pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu116/torch_sparse-0.6.15%2Bpt112cu116-cp39-cp39-linux_x86_64.whl

pip install freerec==0.4.3

echo "Environment setup completed."