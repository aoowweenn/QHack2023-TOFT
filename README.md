# QHack2023-TOFT
Repo for QHack2023 Open Hackathon

## Setup Environments
```bash
virtualenv --python=3.9 env
source env/bin/activate
pip install -r requirements.txt

# directly use cpu
pip install -r requirements_cpu.txt

# try gpu below
pip install cuquantum-cu11

pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install pennylane-lightning[gpu] 

# end gpu

# register this env to jupyter server
python -m ipykernel install --user --name qchem --display-name "QChem"

# jupyter-lab --no-browser
```

## MISC
### If you encounter library error about cudnn runtime version
```bash
pip install nvidia-cudnn-cu11
vim ~/.local/share/jupyter/kernels/qchem/kernel.json
# add 
"env": {"LD_LIBRARY_PATH": "/PathToThisProject/env/lib/python3.9/site-packages/nvidia/cudnn/lib"},
```
