# $\rho$-Diffusion: A diffusion-based density estimation framework for computational physics

<div align="center">

[![workshop-paper](https://img.shields.io/badge/NeurIPS-RhoDiffusion-blue)](https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_71.pdf)
[![lightning](https://img.shields.io/badge/Lightning-v1.8.6%2B-792ee5?logo=pytorchlightning)](https://lightning.ai/docs/pytorch/1.8.6)
[![pytorch](https://img.shields.io/badge/PyTorch-v2.0.1%2B-red?logo=pytorch)](https://pytorch.org/get-started/locally/)
[![ipex](https://img.shields.io/badge/Intel_Extension_For_PyTorch-blue?logo=intel)](https://github.com/intel/intel-extension-for-pytorch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

In computational physics, a classical task is to evaluate the density function as a function of the physics parameters of interest or the model parameters. As the number of parameters increases, the parameter space expands geometrically, which, at some point, renders the evaluation of the density function computationally infeasible.

Instead of resorting to a brute-force evaluation of the density field, `Rho-Diffusion` is a generative model that can learn from existing data and generate the density function based on the parameters it is conditioned. It is based on the Denoising Diffusion Probabilistic Models (DDPMs), which are known for their capability of capturing high-fidelity image data. `Rho-Diffusion` generates the dimensionality of the data to `n`, with a current limitation of `n = {1, 2, 3}`.


## Installation

Use the provided `conda.yml` to create the minimal PyTorch environment: it uses the `intel-aikit-pytorch` package, which includes XPU support for PyTorch 2.0.100.

```console
conda env create -f conda.yml -n rho_diffusion
```

And then install the package with 

```console
pip install .
```
For developers, performing

```console
pip install './[dev]'
```
will install additional developer dependencies. To ensure code consistency, we also use `pre-commit` hooks, which can be set up with `pre-commit install`. The `pre-commit` hooks will create a virtual environment with the developer tools, and run checks with every commit you make to ensure best practices.

## Model Training and Inference
We provide two scripts for easing the model training and inference efforts. These scripts can be found in the `scripts` directory.

### Training
```console
python training.py CONFIG.json
```
`CONFIG.json` is a JSON config file that defines various parameters of the model and the training/inference process. Examples of `CONFIG.json` can be found in the `examples` directory.

### Inference
In the context of DDPM, inference is essentially a process of sampling from a pretrained DDPM model. With `Rho-Diffusion` this can be done by passing the model checkpoint and the config file to the `infernece.py` script:
```console
python inference.py -p CHECKPOINT.pth CONFIG.json
```

## Development
This is a proof-of-concept project under active development. Contributions are much appreciated. Please check out `CONTRIBUTING.md` for more information.

## Citation
If you use `Rho-Diffusion` in your work, please consider cite the following paper:

- Cai, Maxwell X., Lee, Kin Long Kelvin, "Rho-Diffusion: A diffusion-based density estimation framework for computational physics", NeurIPS 2023 Workshop on Machine Learning for Physical Sciences

Please refer to `CITATION.cff` for citing the code.
