!#/bin/bash

conda env create -f environment.yaml
conda activate mesh_funcspace

wget https://download.pytorch.org/whl/cu100/torch-1.0.0-cp37-cp37m-linux_x86_64.whl
wget https://download.pytorch.org/whl/torchvision-0.2.0-py2.py3-none-any.whl

pip install torch-1.0.0-cp37-cp37m-linux_x86_64.whl
pip install torchvision-0.2.0-py2.py3-none-any.whl

python setup.py build_ext --inplace

pip install -U ipykernel