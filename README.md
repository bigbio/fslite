[![Python application](https://github.com/bigbio/fslite/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/enriquea/fslite/actions/workflows/python-app.yml)
[![Python Package using Conda](https://github.com/bigbio/fslite/actions/workflows/python-package-conda.yml/badge.svg?branch=main)](https://github.com/bigbio/fslite/actions/workflows/python-package-conda.yml)

# fslite

---

### Memory-Efficient, High-Performance Feature Selection Library for Big and Small Datasets

### Description

`fslite` is a python module to perform feature selection and machine learning using pre-built FS pipelines. 
Pipelines written using `fslite` can be divided roughly in four major stages: 1) data pre-processing, 2) univariate 
filters, 3) multivariate filters and 4) machine learning wrapped with cross-validation (**Figure 1**).

`fslite` is based on our previous work [feseR](https://github.com/enriquea/feseR); previously implemented in R and caret package; publication can be found [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0189875).

![Feature Selection flowchart](images/fs_workflow.png)
**Figure 1**. Feature selection workflow example implemented in fslite.

### Documentation

The package documentation describes the [data structures](docs/README.data.md) and 
[features selection methods](docs/README.methods.md) implemented in `fslite`.

### Installation

- pip
```bash
git clone https://github.com/bigbio/fslite.git
cd fslite
pip install . -r requirements.txt
```

- conda
```bash
git clone https://github.com/bigbio/fslite.git
cd fslite
conda env create -f environment.yml
conda activate fslite-venv
pip install . -r requirements.txt
```

### Maintainers
- Enrique Audain (https://github.com/enriquea)
- Yasset Perez-Riverol (https://github.com/ypriverol)
