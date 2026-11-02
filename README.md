# scDQ-path

## Algorithm

- **Latent Space Representation:** Compresses high-dimensional single-cell transcriptomic profiles into a 16-dimensional continuous latent space using a Variational Autoencoder (VAE), preserving major transcriptional patterns while reducing dimensionality.
- **DQN-based State Transition:** Employs Double Deep Q-Networks (Double DQN) to learn reward-guided transitions between cellular states, enabling goal-directed exploration of the latent space.
- **Virtual Trajectory Inference:** Explores optimal and alternative latent trajectories to characterize potential cellular state transitions beyond conventional pseudotime-based approaches.
- **Dynamic Transcription Factor Analysis:** Identifies transcription factors exhibiting dynamic expression patterns along inferred trajectories to investigate potential regulatory changes associated with cellular state transitions.
- **Counterfactual Gene Knockout Simulation:** Performs in silico gene knockout simulations through latent-space optimization while keeping the trained VAE decoder frozen, allowing transcriptional responses to be investigated under predefined perturbation conditions.
- **Gene Regulatory Network Construction:** Constructs directed perturbation-response networks by evaluating transcriptional changes associated with simulated gene perturbations and their downstream regulatory effects.
- **Reverse scDQ-path:** Extends trajectory exploration through multiple DQN agents and latent-space visitation-density analysis, enabling probabilistic sampling of virtual cells from explored cellular state distributions.
- **Multi-target Perturbation:** Supports both single-target and multi-target knockout simulations to investigate individual and combined perturbation-associated cellular responses.

## Key Novelty

- **RL-driven Latent Space Exploration:** Introduces reinforcement learning as an approach to exploring continuous single-cell latent spaces, moving beyond static cell-state representations and conventional pseudotime ordering.
- **Alternative and Compensatory Dynamics:** Captures multiple possible cellular trajectories, including alternative and potentially compensatory response paths, rather than restricting inference to a single progression route.
- **Counterfactual Perturbation Modeling:** Integrates latent-space optimization with a frozen generative decoder to investigate transcriptional changes under simulated gene knockout conditions.
- **Density-guided Virtual Cell Generation:** Introduces Reverse scDQ-path to aggregate latent-space exploration patterns and perform probabilistic sampling, generating heterogeneous virtual cell populations.
- **Unified Dynamics and Perturbation Framework:** Integrates trajectory inference, regulatory analysis, counterfactual knockout simulation, and virtual cell generation within a unified reinforcement learning framework.

## Workflow

| Step | Process | Description |
|:---:|---|---|
| **01** | [**Data Preprocessing (PBMC)**](https://github.com/AiJayce/scDQ-path/tree/Data-Preprocessing-%28PBMC%29) | Preprocessing, quality control, normalization, and feature selection of PBMC single-cell transcriptomic data. |
| **02** | [**Virtual Trajectory Construction**](https://github.com/AiJayce/scDQ-path/tree/Virtual-Trajectory-Construction) | DQN-based exploration of cellular state transitions and construction of continuous virtual trajectories. |
| **03** | [**Gene Regulatory Network (GRN) Construction**](https://github.com/AiJayce/scDQ-path/tree/Gene-Regulatory-Network-%28GRN%29-Construction) | Inference of directed gene regulatory relationships and analysis of trajectory-dependent transcriptional dynamics. |
| **04** | [**Reverse scDQ-path: Virtual Cell Sampling**](https://github.com/AiJayce/scDQ-path/tree/Reverse-scDQ-path-Virtual-Cell-Sampling) | Reverse latent-space exploration and density-guided probabilistic sampling to generate perturbation-associated virtual cells. |
| **05** | [**Benchmark and Validation**](https://github.com/AiJayce/scDQ-path/tree/Benchmark-and-Validation) | Quantitative evaluation of virtual cell generation and perturbation responses against observed single-cell transcriptomic profiles. |

## Model Strengths

- **Agent-based Continuous Trajectory Learning:** Utilizes trained DQN agents to learn sequential state transitions in a continuous latent space, enabling flexible trajectory inference beyond predefined cell-state ordering.
- **Directional Network Analysis:** Infers directed regulatory relationships and distinguishes positive (+) and negative (-) regulatory effects based on simulated transcriptional responses.
- **Sparse and Non-sparse Network Analysis:** Supports both sparse and non-sparse network representations, allowing regulatory interactions to be investigated under different network constraints.
- **Network Power Estimation:** Evaluates the magnitude of regulatory influence through network power analysis, providing an additional measure of the potential impact of individual regulatory interactions.
- **Probabilistic Virtual Cell Sampling:** Enables probabilistic sampling of heterogeneous virtual cell populations following gene knockout simulations by leveraging explored latent-space distributions and visitation densities.

----
For a more detailed explanation of the scDQ-path model, architecture, and algorithms, [**Click Here**](https://drive.google.com/file/d/1GaReSXWjaVRdAZodmbqQbnlcIYXwr0z8/view?usp=drive_link).
----

## Core Technologies

`Python` · `PyTorch` · `Variational Autoencoder (VAE)` · `Double DQN` · `Reinforcement Learning` · `Latent Space Exploration` · `Gene Regulatory Network (GRN)` · `Counterfactual Simulation` · `Virtual Cell` · `scRNA-seq`

