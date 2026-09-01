# LLaMA-Factory environment

The current training environment uses Python 3.11, PyTorch 2.13.0 with CUDA
13.0, Transformers 5.2.0, and LLaMA-Factory 0.9.5.dev0 from this repository.

Create a separate reproducible environment without changing the active one:

```bash
bash scripts/create_llamafactory_env.sh llamafactory-repro
conda activate llamafactory-repro
```

Validate the current environment:

```bash
conda run -n llamafactory python scripts/verify_llamafactory_env.py
```

`llamafactory-core.yml` contains the maintained training stack. The lock file
is an exact snapshot of all currently installed Python packages and is intended
for auditing or exact-version troubleshooting rather than routine installation.

The setup script also configures `CUDA_HOME` to the CUDA 13 toolkit installed
inside the new environment. Always activate the Conda environment (or use
`conda run`) so this variable is applied before importing DeepSpeed.
