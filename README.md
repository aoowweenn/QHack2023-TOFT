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
pip install nvidia-cudnn-cu11
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install pennylane-lightning[gpu] 

# register this env to jupyter server
python -m ipykernel install --user --name qchem --display-name "QChem"

# jupyter-lab --no-browser
```
