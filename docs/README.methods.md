
# fsspark - features selection methods 

---

`fsspark `includes a set of methods to perform feature selection and machine learning based on spark.
A typical workflow written using `fsspark` can be divided roughly in four major stages:

1) data pre-processing. 
2) univariate filters. 
3) multivariate filters.
4) machine learning wrapped with cross-validation.

### 1. Data pre-processing

- a) Filtering by missingness rate. 
   - Remove features from dataset with high missingness rates across samples.
- b) Impute missing values.
     - Impute features missing values using mean, median or mode.
- c) Scale features.
   - Normalize features using MinMax, MaxAbs, Standard or Robust methods.

### 2. Available univariate filters

- a) Univariate correlation
  - Compute correlation between features and a target response variable and keep
    uncorrelated features.
- b) Anova test
  - Select features based on an Anova test between features and a target response 
    variable (categorical).
- c) F-regression
  - Select features based on a F-regression test between features and a target response 
    variable (continuous).

### 3. Available multivariate filters

- a) Multivariate correlation
  - Compute pair-wise correlation between features and remove highly correlated features.
- b) Variance
  - Remove features with low-variance across samples.

### 4. Machine-learning methods with cross-validation

- a) Random Forest classification
  - For classification tasks on both binary and multi-class response variable (categorical).
- b) Linear Support Vector Machine
  - For classification tasks on binary response variable.
- c) Random Forest regression
  - For regression tasks (e.g. continuous response variable).
- d) Factorization Machines
  - For regression problems (e.g. continuous response variable).


### 5. Feature selection pipeline example

[FS pipeline example](../fsspark/pipeline/fs_pipeline_example.py)
