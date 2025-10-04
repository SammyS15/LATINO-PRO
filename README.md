# Official repository for **LATINO-PRO**

> **LAtent consisTency INverse sOlver with PRompt Optimization** – <https://arxiv.org/abs/2503.12615>.

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=<USERNAME>.<REPO>)
[![GitHub stars](https://img.shields.io/github/stars/LATINO-PRO/LATINO-PRO.svg?style=social&label=Stars)](https://github.com/<USERNAME>/<REPO>/stargazers)
---

## 📦 Installation

```bash
# 1. Clone the repo and enter it
git clone git@github.com:LATINO-PRO/LATINO-PRO.git
cd LATINO-PRO

# 2. (Optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# Windows-PowerShell:  .venv\Scripts\Activate.ps1

# 3. Install all Python dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

No custom CUDA extensions are required; GPU acceleration is handled automatically by PyTorch if a compatible device is present.

---

## 🚀 Quick start

The repository contains two ready‑to‑run scripts. **All hyper‑parameters are controlled by the YAML files inside the **configs** directory**, so the basic usage is simply:

```bash
# Baseline LATINO model
python main_LATINO.py            # uses configs/LATINO.yaml by default

# Prompt‑optimized LATINO‑PRO model
python main_LATINO_PRO.py        # uses configs/LATINO_PRO.yaml by default
```

```configs/problem``` offers different inverse problem operators to choose from.

```configs/image``` includes two examples taken from the FFHQ and AFHQ datasets.

---

## 📓 Interactive notebooks

| Notebook           | Purpose                                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| LATINO.ipynb       | Hands‑on introduction to the baseline solver: load a sample, apply degradation, reconstruct, inspect metrics. |
| LATINO_PRO.ipynb   | Full prompt‑optimization workflow adapted to the LoRA-LCM model to work on small GPUs (eg. Colab T4 GPU).     |

---

## 🗂️ Repository layout

```
LATINO-PRO/
├── configs/              # YAML config files controlling every experiment
├── samples/              # example images for tests
├── LATINO.ipynb          # baseline interactive notebook
├── LATINO_PRO.ipynb      # full prompt‑optimization notebook
├── LICENSE               # License
├── inverse_problems.py   # deepinverse operators are defined here
├── main_LATINO.py        # base LATINO restoration
├── main_LATINO_PRO.py    # prompt‑optimized restoration
├── motionblur.py         # helper code for motion‑blur degradations
├── noise_schemes.py      # definition of various inverse solvers
├── utils.py              # miscellaneous utilities
├── requirements.txt      # Python dependencies
└── README.md             # this file
```

---

## 📄 Citation

If you use LATINO‑PRO in academic work, please cite:

```bibtex
@misc{spagnoletti2025latinoprolatentconsistencyinverse,
      title={LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization}, 
      author={Alessio Spagnoletti and Jean Prost and Andrés Almansa and Nicolas Papadakis and Marcelo Pereyra},
      year={2025},
      eprint={2503.12615},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.12615}, 
}
```

---

## 🛡️ License

Distributed under the **MIT License**. See `LICENSE` for more information.

---
