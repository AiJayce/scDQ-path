# Benchmark-and-Validation

## 5-1. Analysis of ATF6 Suppression Trajectories Using Perturb-seq Data

### Introduction

To apply the latent space-based trajectory exploration method developed during the PBMC analysis to actual gene perturbation data, the K562 Perturb-seq dataset from Adamson et al. (2016) was used.

This dataset combines CRISPR interference (CRISPRi)-based gene suppression with single-cell RNA sequencing to measure transcriptomic responses following the suppression of specific genes. In this analysis, the negative control and ATF6-targeting conditions were selected to generate virtual trajectories associated with reduced ATF6 expression and compare the reconstructed expression profiles with those of experimentally measured ATF6-perturbed cells.

ATF6 is a transcription factor involved in endoplasmic reticulum stress and the unfolded protein response (UPR). Therefore, ATF6 suppression was used as a benchmark case to evaluate how well the model reproduces not only the reduction of the target gene but also the associated transcriptomic changes.

### Dataset Information

The benchmark dataset was obtained from the Perturb-seq data published by Adamson et al. (2016) in the NCBI Gene Expression Omnibus (GEO). The AnnData-formatted distribution provided through scPerturb was used for this analysis.

| Attribute | Description |
|---|---|
| GEO Accession | GSE90546 |
| GEO Sample | GSM2406677 |
| Sample Title | Epistasis experiment cells |
| Organism | Homo sapiens |
| Cell Line | K562 |
| Experimental Platform | 10x Genomics Chromium-based single-cell RNA-seq |
| Perturbation Method | CRISPR interference (CRISPRi) |
| Selected Conditions | Negative control and ATF6-targeting condition |
| Processed Data | scPerturb / Zenodo distribution |
| Reference | Adamson et al., Cell, 2016 |

The following two conditions were selected for preprocessing and subsequent analysis.

| GEO Sample ID | Condition Label | Group |
|---|---|---|
| GSM2406677 | 3x_neg_ctrl_pMJ144-1 | Negative control |
| GSM2406677 | ATF6_only_pMJ145 | ATF6-targeting CRISPRi |

These two conditions are perturbation labels within the same GEO sample, not separate GEO samples. In addition, since the actual experiment used CRISPRi-mediated suppression, the ATF6-zero condition used in the model should not be interpreted as equivalent to complete gene deletion.

### Preprocessing and Gene Selection

Raw counts were preserved, followed by cell-wise CP10K normalization across all genes and logarithmic transformation.

$$
x_{cg}=
\log\left(
1+10^4\frac{n_{cg}}{\sum_h n_{ch}}
\right)
$$

Genes were ranked based on expression differences between the control and ATF6-targeting conditions. The required genes ATF6, HSPA5, HSP90B1, HERPUD1, PDIA4, and CALR were explicitly included as ATF6- and UPR-related genes.

The currently saved trajectory model uses 3,000 genes and a 128-dimensional latent space. For expression comparisons across tools, a separate fixed evaluation panel of 2,000 genes was used. This evaluation panel consists of 1,994 non-required genes selected according to signed log fold-change ranking and six required genes.

No p-value cutoff was applied during gene selection. Therefore, the selected genes should not be interpreted as a set in which every gene is a statistically significant differentially expressed gene (DEG).

### ATF6-Decreasing Virtual Pathway Exploration

<img width="3456" height="2030" alt="5-1" src="https://github.com/user-attachments/assets/4fcc0efd-8b49-4593-ab96-430b305d41ab" />

The median control coordinate across each PCA dimension was inverse-transformed through the existing PCA model to establish a common starting coordinate in the fixed VAE latent space. The model then explored trajectories associated with reduced ATF6 expression.

Unlike target-directed trajectory modeling, neither a specific ATF6-positive cell nor a destination coordinate was specified. Instead, the reduction of ATF6 expression itself was used as the exploration objective.

| Component | Implementation | Purpose |
|---|---|---|
| Starting point | Virtual latent coordinate obtained by inverse PCA transformation of the dimension-wise median control PCA coordinates | Standardize the initial state across all simulations |
| Computational space | 128-dimensional VAE latent space | Explore a compressed representation of high-dimensional expression data |
| Actions | 256 actions representing positive and negative movements along each latent axis | Explore different directions of expression change |
| Step size | 0.01 times the observed standard deviation of the corresponding latent dimension | Construct trajectories through accumulated small movements |
| Reward | Action-specific ATF6 reduction relative to the initial ATF6 expression | Optimize target expression reduction |
| Constraints | Only actions remaining within the empirical cell support and producing non-negative, reduced ATF6 expression are allowed | Restrict excessive deviation and movements that contradict the objective |
| Trajectory generation | 48 simulations generated using four independent DQN models | Capture trajectory diversity arising from different policies and exploration |
| Branch analysis | Ward clustering based on net latent displacement across three trajectory segments | Summarize similar movement patterns into representative branches |
| Visualization | Project trajectories onto the same fixed UMAP embedding | Compare observed cells and generated trajectories within a common coordinate system |

The reward is defined as:

$$
r_t=
\frac{ATF6_t-ATF6_{t+1}}
{ATF6_{\mathrm{start}}}
$$

Entropy is used to monitor changes in the expression distribution and is not included in the reward. UMAP is used exclusively for visualization, while action selection and empirical support constraints are evaluated in the latent space.

In the attached simulation output, the 48 trajectories were classified into three branches.

| Branch | Number of trajectories | Representative trajectory steps | Initial ATF6 | Final ATF6 |
|---|---:|---:|---:|---:|
| 1 | 20 | 401 | 0.2305 | 0.1368 |
| 2 | 17 | 500 | 0.2305 | 0.1281 |
| 3 | 11 | 500 | 0.2305 | 0.1198 |

All three branches showed reduced ATF6 expression, but differed in their movement patterns and final expression levels. The density panel on the right represents the relative density of post-movement coordinates visited by all simulations, rather than the density of actual cells.

---

## 5-2. Perturbation Prediction Benchmark

### Benchmark Design

<img width="3669" height="1766" alt="5-2" src="https://github.com/user-attachments/assets/30f6a234-eac5-4241-9e9c-3d8f19acea51" />

Virtual ATF6 perturbation expression generated by scDQ-path was compared with experimentally measured expression. The prediction results from scGen, CPA, and CellOracle were also evaluated.

Two simple baselines were additionally included to determine whether the model could reproduce transcriptomic responses beyond simply setting the target gene to zero.

All methods were evaluated using the same 2,000-gene order and log1p CP10K expression scale. Normalization was performed across all genes before extracting the evaluation panel; no additional normalization was performed within the evaluation panel.

### Data Splitting

The 1,715 control cells and 1,202 ATF6-targeting cells used for comparison were divided into approximately 70/10/20% splits within each condition.

| Split | Control | ATF6 perturbation | Purpose |
|---|---:|---:|---|
| Train | 1,200 | 840 | Model training |
| Validation | 172 | 121 | Validation during training |
| Test | 343 | 241 | Predict from control inputs and compare against actual ATF6-perturbed cells |

External tools generated predictions using the same 343 test control cells and were evaluated against the same 241 actual ATF6 test cells.

For scDQ-path, 821, 326, and 55 virtual cells generated from three visitation-density intervals were combined, resulting in 1,202 virtual cells. These were compared against the same 241 actual ATF6 test cells. The evaluation was performed using all generated virtual cells combined, rather than an equally weighted average of scores calculated separately for each density interval.

**Difference in information conditions:** The representation learning used by scDQ-path also included information from the evaluation KO cells. Therefore, scDQ-path is presented as a KO-informed reference in the figures and should be distinguished from the independent held-out prediction results of the external tools.

### Compared Methods

| Method | Main characteristics | Information used in this comparison |
|---|---|---|
| Control / no change | Retains the original control expression without perturbation | Baseline without perturbation prediction |
| ATF6 zero only | Sets only ATF6 expression to zero in control cells | Baseline applying only target gene suppression |
| scGen | Predicts expression using the control-to-perturbation difference in VAE latent space | Training control and ATF6-perturbed cells |
| CPA | Models expression responses by separating cell state and perturbation effects | Training control and ATF6-perturbed cells |
| CellOracle | Infers a GRN and propagates target suppression effects through the network | Training control cells and a public human promoter GRN |
| scDQ-path | Generates expression profiles using the VAE decoder by sampling around DQN trajectories | KO-informed reference using control and ATF6 information |

scGen and CPA perform held-out cell prediction for an ATF6 perturbation that was already included during model training. CellOracle uses a control-only approach that does not train on KO expression profiles, resulting in a different information condition.

This benchmark represents a comparison under the fixed configurations used in the current execution, rather than an optimized performance comparison involving extensive hyperparameter searches.

### Evaluation Metrics

Let $\hat{x}_{ig}$ denote the predicted expression of gene $g$ in predicted cell $i$, and $y_{jg}$ denote the expression of gene $g$ in actual ATF6-perturbed cell $j$. The number of evaluation genes is defined as $G=2000$.

#### 1. Cell-wise Cosine Similarity

$$
C_{ij}=
\frac{\sum_g \hat{x}_{ig}y_{jg}}
{\sqrt{\sum_g\hat{x}_{ig}^2}
\sqrt{\sum_g y_{jg}^2}}
$$

Expression-vector directional similarity was calculated across all combinations of predicted cells and actual ATF6-perturbed cells. Higher values indicate greater similarity in the overall expression patterns.

This metric does not evaluate the accuracy of changes in one-to-one corresponding cells. Since multiple cell pairs share the same cells, individual pairwise values should not be treated as statistically independent observations.

#### 2. Gene-wise Mean Residual

$$
e_g = \mathrm{mean}(\hat{x}_{ig}) - \mathrm{mean}(y_{jg})
$$

This metric represents the difference between the predicted and actual mean expression of each gene.

- $e_g>0$: Overestimation of mean expression
- $e_g<0$: Underestimation of mean expression
- $e_g$ close to 0: Mean expression is similar to the actual value

A median residual close to zero may conceal large errors in individual genes. Therefore, MAE and RMSE were also evaluated.

#### 3. Pearson Correlation

$$
r=
\frac{
\sum_g(\bar{\hat{x}}_g-\bar{\hat{x}})
(\bar{y}_g-\bar{y})
}{
\sqrt{\sum_g(\bar{\hat{x}}_g-\bar{\hat{x}})^2}
\sqrt{\sum_g(\bar{y}_g-\bar{y})^2}
}
$$

This metric measures the correlation between the mean expression vectors of the 2,000 genes. Higher values indicate greater similarity in the relative mean expression patterns across genes. However, a high correlation can still occur when the overall expression magnitude is systematically biased.

#### 4. MAE and RMSE

$$
MAE=\frac{1}{G}\sum_g|e_g|
$$

$$
RMSE=\sqrt{\frac{1}{G}\sum_g e_g^2}
$$

| Metric | Description | Preferred direction |
|---|---|---|
| MAE | Absolute magnitude of gene-wise mean expression errors | Lower is better |
| RMSE | Mean error that assigns greater weight to large deviations | Lower is better |

Here, MAE and RMSE are calculated from gene-wise mean expression residuals, rather than errors between individually matched cells.

#### 5. Cosine Similarity Distribution

The cosine similarity values calculated in metric 1 were visualized using histograms with common bins. This panel was used to examine the concentration of values in the high-similarity range and the presence of lower-similarity tails. It does not represent an additional independent evaluation metric.

### Benchmark Results

| Method | Cosine Median ↑ | Pearson ↑ | MAE ↓ | RMSE ↓ |
|---|---:|---:|---:|---:|
| Control / no change | 0.8722 | 0.9949 | 0.001697 | 0.015200 |
| ATF6 zero only | 0.8759 | 0.9962 | 0.001619 | 0.014581 |
| scGen | 0.9324 | **0.9982** | 0.001698 | **0.004905** |
| CPA | 0.9271 | 0.9956 | 0.005222 | 0.018784 |
| CellOracle | 0.8964 | 0.9960 | 0.001657 | 0.014730 |
| scDQ-path — KO-informed reference | **0.9353** | 0.9980 | **0.001550** | 0.005871 |

Bold values indicate the numerically best results among all displayed methods, including reference groups. They do not establish superiority under equivalent information conditions.

### Strengths of scDQ-path

#### 1. High Overall Expression Pattern Similarity

scDQ-path achieved a cosine similarity median of 0.9353, the highest value among the methods presented in this comparison. This exceeded the Control baseline (0.8722) and the ATF6-zero-only baseline (0.8759), indicating that the model reconstructed expression patterns involving genes beyond the target gene itself.

However, this result represents a KO-informed reconstruction and should not be interpreted as predictive performance on an unseen perturbation.

#### 2. Lowest Mean Absolute Error

The MAE of gene-wise mean expression was 0.001550, approximately 8.7% lower than that of scGen. This indicates favorable performance not only in overall expression similarity but also in the absolute error of mean expression.

However, scGen achieved a lower RMSE. Therefore, scDQ-path did not outperform all methods across every error metric.

#### 3. Joint Representation of Expression Prediction and Trajectory Interpretation

In addition to the final expression matrix, scDQ-path provides action-specific target expression changes, multiple movement trajectories, representative branches, and visitation density.

This enables exploration not only of endpoint agreement but also of the intermediate states through which the model generates expression changes.

### Overall Interpretation

In this benchmark, scDQ-path demonstrated high cosine similarity and low MAE as a KO-informed reconstruction approach, while also providing interpretable trajectories and branching patterns. Its main contribution is the ability to examine latent movement patterns and intermediate states alongside the reconstructed perturbation expression profile.

Nevertheless, the inclusion of KO information during representation learning limits direct comparison with independent held-out prediction methods. The current results therefore support the use of scDQ-path as a trajectory-based reconstruction and interpretation framework, rather than establishing superior generalization to unseen perturbations.

----
[Click here to view the Benchmark and Validation notebook](https://github.com/AiJayce/scDQ-path/blob/Benchmark-and-Validation/Total_benchmark.ipynb)
----

