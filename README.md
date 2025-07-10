# Filter-based System Level Tube-MPC
[GitHub](https://github.com/IntelligentControlSystems/filter-sltmpc) | [Paper](https://www.sciencedirect.com/science/article/pii/S0005109825003607) | [Issues](https://github.com/IntelligentControlSystems/filter-sltmpc/issues)

``filter-sltmpc`` -- *Code accompanying the paper:*

J. Sieber, A. Didier, and M. N. Zeiligner, "[Computationally efficient system level tube-MPC for uncertain systems](https://www.sciencedirect.com/science/article/pii/S0005109825003607)", Automatica, 2025 (open access).

The extended version of the paper is available on [arXiv](https://arxiv.org/abs/2406.12573).

The proposed filter-based system level tube-MPC (SLTMPC) method is a more general and recursively feasible version of SLS-MPC proposed in [1]. It is designed for systems with model uncertainties and additive disturbances. The main idea is to overapproximate the combined uncertainties with an online optimized set, while also optimizing the tube controller online. This results in a general and nonconservative MPC method.

[1] S. Chen, V. M. Preciado, M. Morari, and N. Matni, "[Robust Model Predictive Control with Polytopic Model Uncertainty through System Level Synthesis](https://www.sciencedirect.com/science/article/abs/pii/S0005109823005988)", Automatica, 2024.

## Installation

``filter-sltmpc`` requires Python 3.10 or higher and depends on the ``ampyc`` [package](https://github.com/IntelligentControlSystems/ampyc).

For the setup, clone this repository and install ``ampyc`` using pip, i.e.
```
    python3 -m pip install ampyc
```

Alternatively, you can install all requirements using the provided `requirements.txt` file:
```
    python3 -m pip install -r requirements.txt
```
## Getting Started
To get started with ``filter-sltmpc`` package, run [this notebook](https://github.com/IntelligentControlSystems/filter-sltmpc/blob/main/filter_sltmpc.ipynb) after installation. The notebook highlights the main features of the proposed MPC method and compares it to SLS-MPC [1].

## Citation
If you find this method helpful, please cite our work:
```bib
@article{Sieber2025,
title = {Computationally efficient system level tube-{MPC} for uncertain systems},
journal = {Automatica},
volume = {180},
pages = {112466},
year = {2025},
issn = {0005-1098},
doi = {https://doi.org/10.1016/j.automatica.2025.112466},
url = {https://www.sciencedirect.com/science/article/pii/S0005109825003607},
author = {Jerome Sieber and Alexandre Didier and Melanie N. Zeilinger},
}
```
