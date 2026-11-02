# Before Getting Started

## Why Careful Data Preprocessing Matters

The quality of downstream single-cell analysis fundamentally depends on the quality and consistency of the input data. In single-cell RNA sequencing (scRNA-seq), observed gene expression profiles contain not only biological variation associated with cellular identity and functional states, but also technical variation introduced during sample preparation, sequencing, and library construction.

Without appropriate preprocessing, technical artifacts such as low-quality cells, sequencing depth differences, mitochondrial RNA contamination, and sample-specific batch effects can become confounded with genuine biological signals. These artifacts may distort cellular similarity, alter clustering structures, and introduce misleading patterns into downstream analyses.

This issue is particularly important for computational frameworks such as scDQ-path, where cellular relationships and state transitions are explored through latent representations and reinforcement learning. Since trajectory construction, gene regulatory network (GRN) inference, and virtual cell generation rely on the underlying structure of the expression space, preprocessing errors can propagate throughout the entire analytical pipeline.

For example, insufficient quality control may introduce damaged or transcriptionally inactive cells, while inadequate normalization can cause sequencing depth to dominate cellular distances. Similarly, unresolved batch effects may lead models to interpret technical differences between samples as biological state transitions. Conversely, excessive batch correction may remove genuine biological variation, potentially obscuring meaningful cellular heterogeneity.

Therefore, preprocessing is not merely a preliminary data-cleaning step. It establishes a reliable representation of biological variation, allowing subsequent computational models to learn cellular relationships and transcriptional dynamics from biologically meaningful signals rather than technical artifacts.

The preprocessing workflow implemented in this project consists of quality assessment, feature selection, dimensionality reduction, sample integration, and hierarchical cell annotation.

## Dataset Information

The single-cell RNA sequencing dataset used in this project was obtained from the NCBI Gene Expression Omnibus (GEO).

- **GEO Accession:** [GSE316482](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE316482)
- **Selected Samples:** Three healthy control (HC) samples
- **Sample Selection:** HC01, HC02, and HC03

The following three samples were selected for downstream preprocessing and analysis.

| GEO Sample ID | Sample Label | Group |
|:---:|:---:|:---:|
| GSM9453841 | HC01 | Healthy Control |
| GSM9453842 | HC02 | Healthy Control |
| GSM9453843 | HC03 | Healthy Control |

Only these three selected samples were included in the preprocessing workflow described below.

## 1. Data Preprocessing

### 1-1. Cell Quality Assessment (Quality Control)

<img width="3731" height="1493" alt="1-1" src="https://github.com/user-attachments/assets/12d37117-2fb6-453e-93a5-989035d91242" />

Cell quality was assessed by visualizing the total number of UMIs, the number of detected genes, and the proportion of mitochondrial gene expression for each cell. These metrics provide complementary information for identifying cells with insufficient RNA detection, potential cellular damage, or abnormally high expression levels that may indicate technical artifacts such as doublets.

Scatter plots and violin plots were jointly examined to evaluate the overall distributions, identify outliers, and assess potential technical biases that could influence subsequent analyses. This process establishes a quality-controlled cellular population while minimizing the introduction of artifacts into downstream modeling.

### 1-2. Highly Variable Gene Selection and PCA-Based Data Exploration

<img width="3981" height="1472" alt="1-2" src="https://github.com/user-attachments/assets/fa377dff-eac3-4c47-afae-6384db331471" />

Following expression normalization and logarithmic transformation, highly variable genes (HVGs) were identified to prioritize genes exhibiting substantial expression variability across cells.

Rather than using all detected genes indiscriminately, feature selection focuses subsequent dimensionality reduction on genes that capture informative differences in cellular identity and transcriptional states. This helps reduce the influence of less informative features and limits computational complexity.

Principal Component Analysis (PCA) was subsequently performed to investigate the major sources of variation in the expression matrix. The explained variance and cumulative explained variance of individual principal components were examined to characterize the underlying data structure and determine an appropriate dimensionality for subsequent analyses.

### 1-3. scVI-Based Sample Integration and Latent Space Visualization

<img width="3996" height="1302" alt="1-3" src="https://github.com/user-attachments/assets/6033ed80-5e28-4fbe-8c72-a9fd2f8b184b" />

To minimize technical variation between samples, a scVI model was trained using sample identities as batch covariates.

scVI uses a probabilistic deep generative framework to learn a low-dimensional latent representation of single-cell gene expression profiles while accounting for sample-associated technical variation.

The resulting latent representations were used to construct a neighborhood graph and generate a UMAP embedding. Training and validation loss curves were examined to assess model convergence and learning behavior.

Additionally, sample-specific UMAP visualizations were used to evaluate the degree of inter-sample mixing and identify regions potentially dominated by individual samples. This step helps assess whether the learned latent representation captures shared cellular structures rather than being primarily driven by sample-specific technical differences.

### 1-4. Automated Broad Cell Annotation Using CellTypist

<img width="3848" height="1608" alt="1-4" src="https://github.com/user-attachments/assets/efb0b54a-bf86-4b4e-a4f6-31a1381a8142" />

To characterize the major cellular composition of PBMC samples, automated cell annotation was performed using CellTypist with a pretrained PBMC reference model.

This approach enables broad classification of major immune cell populations, including T cells, B cells, NK cells, monocytes, and dendritic cells, providing an initial annotation framework for subsequent cell-type-specific analyses.

Cells with insufficient prediction confidence were excluded based on prediction scores. The resulting annotations were further evaluated using UMAP visualizations and confidence score distributions to examine the spatial consistency and reliability of the predicted cell identities.

### 1-5. T Cell Subsetting and Marker-Based Fine Annotation

<img width="2734" height="2228" alt="1-5" src="https://github.com/user-attachments/assets/378ebb87-de40-4129-b058-697e970c9a8e" />

Following broad cell annotation, CD4 and CD8 T-cell populations were selected for more detailed investigation. The selected T cells were subsequently reprocessed and reclustered to resolve transcriptional heterogeneity that could otherwise be obscured by large-scale differences between major immune cell populations.

Differentially expressed genes across individual clusters were examined alongside established cell-type-specific markers to perform fine-grained annotation.

The resulting populations included naive, memory, effector memory, cytotoxic, and regulatory T-cell subsets. Additionally, the selected γδ T-cell lineage was further characterized into γδ T / MAIT-like populations based on their marker expression patterns.

Finally, UMAP visualizations and marker-based dot plots were generated to demonstrate the distribution of individual T-cell subpopulations and provide transcriptional evidence supporting the final annotations.
