#!/bin/bash

pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

pip install torchdata==0.4.1

pip install freerec==0.4.3

echo "Environment setup completed."