# Reverse scDQ path Virtual Cell Sampling

## 4-1. Modeling FOXP3 Reduction Trajectories from a Representative Treg State

<img width="1030" height="480" alt="4-0" src="https://github.com/user-attachments/assets/d321e404-2c50-4f99-bccb-2aa05b51b1a4" />

Starting from a representative latent state of Regulatory T cells, multiple virtual trajectories associated with reduced FOXP3 expression were explored. Unlike the previous Naive-to-Treg target-directed model, this model does not specify a destination coordinate. Instead, it defines only FOXP3 reduction as the expression-level objective. This allows the model to explore different movement patterns that reduce FOXP3 expression from the same initial state, rather than following a trajectory toward a predefined destination.

The existing VAE and PCA models were kept fixed, while the DQN repeatedly performed small movements in the latent space. Gene expression at each new state was evaluated using the fixed decoder, ensuring that the expression model remained unchanged throughout policy learning.

### Model Configuration

| Component | Implementation | Rationale |
|---|---|---|
| Input data | Latent coordinates and cell-type annotations from the existing T-cell VAE | Maintain the same representation space as previous analyses |
| Gene expression features | 2,000 genes learned by the existing VAE | Maintain consistency in gene features and decoder outputs across analyses |
| Expression model | Fixed MLP decoder with an architecture of 16 → 128 → 256 → 2,000 | Evaluate expression at new coordinates using the same model |
| Starting coordinate | Mean of the 16-dimensional latent coordinates of actual Regulatory T cells | Explore trajectories from a representative Treg state rather than individual cells |
| Target coordinate | Not specified | Explore expression-reduction trajectories without enforcing a predefined destination |
| Environment state | Continuous 16-dimensional virtual latent coordinates generated through actions | Explore intermediate states that are not necessarily observed in the original data |
| Policy input | 35 features: 16 standardized coordinates, 16 displacements relative to the starting point, and one feature each for FOXP3 expression, entropy, and distance from the starting point | Provide both the current state and its displacement relative to the initial state |
| Actions | 32 discrete actions representing positive and negative movements along each latent axis | Select discrete movements that increase or decrease individual latent dimensions |
| Step size | 0.01 times the observed standard deviation of the corresponding latent axis | Account for axis-specific scales while maintaining small movements |
| Action constraints | Candidate states must remain within the empirical cell support, have non-negative FOXP3 expression, and show a decrease relative to the current state | Prevent excessive deviation from the observed data distribution and movements that contradict the objective |
| Reward | Action-specific FOXP3 reduction normalized by the starting FOXP3 expression | Directly optimize the reduction in FOXP3 expression |
| Role of entropy | State feature and monitoring metric | Track changes in the expression distribution separately from the reward |
| Q-network | Two hidden layers, each containing 256 ReLU neurons | Estimate the long-term value of actions at each state |
| Training strategy | Double DQN using a replay buffer and target network | Reuse experience and stabilize value learning |
| Independent models | Four DQN models trained using different random seeds | Explore trajectories without relying on a single initialization |
| Final simulation | 12 trajectories per model, resulting in 48 trajectories in total | Capture diversity arising from policy differences and exploratory actions |

### Actions and Reward

Let $\sigma_j$ denote the observed standard deviation of latent dimension $j$. A single movement is defined as:

$$
z_{t+1}=z_t\pm0.01\sigma_j e_j
$$

The reward is defined as the reduction in FOXP3 expression reconstructed by the decoder, denoted by $F_t$:

$$
r_t=\frac{F_t-F_{t+1}}{F_{\mathrm{start}}}
$$

Thus, actions are evaluated according to how much FOXP3 expression decreases relative to its initial level. The current implementation accepts only actions that produce a reduction sufficiently larger than numerical error. Consequently, detour trajectories that temporarily increase FOXP3 expression before achieving a larger subsequent reduction are not explored.

Entropy is calculated from the relative distribution of reconstructed gene expression:

$$
p_g=
\frac{\mathrm{expm1}(\max(f_g(z),0))}
{\sum_h \mathrm{expm1}(\max(f_h(z),0))}
$$

$$
H(z)=-\sum_g p_g\log p_g
$$

In the current implementation, $H$ is not included in the reward, nor is entropy reduction required as a condition for trajectory progression.

### Movement Constraints and Termination Conditions

| Condition | Implementation | Purpose |
|---|---|---|
| Empirical cell support | The distance to the fifth-nearest observed cell must be within the 95th percentile of the reference distance distribution | Restrict excessive deviation from the observed data distribution |
| Exhaustion of FOXP3-reducing actions | Terminate when no valid action producing further FOXP3 reduction remains | Prevent additional movements that cannot satisfy the objective |
| Maximum number of movements | 500 actions | Limit the exploration length |
| Simulation stagnation | Terminate when the standardized net displacement over the most recent 10 movements is 0.04 or less | Detect repetitive movements within a limited region |

Stagnation is determined using the net displacement rather than the cumulative movement distance:

$$
\left|
\frac{z_t-z_{t-10}}{\boldsymbol{\sigma}}
\right|_2
\leq0.04
$$

This stagnation criterion is applied during evaluation and simulation, but not as a terminal condition during training episodes. In addition, a reduction in FOXP3 expression alone is not interpreted as evidence that a trajectory has reached a particular cell type.

---

## 4-2. Branching of Multiple Simulated Trajectories and Representative Path Analysis

<img width="3587" height="2354" alt="4-1" src="https://github.com/user-attachments/assets/3b91782c-93f4-401f-a903-622f19fb8f3a" />

To investigate whether different FOXP3-reduction trajectories emerge from the same Treg starting point, 48 simulations were generated using four independently trained policies. For each model, one greedy trajectory was generated by selecting the action with the highest Q-value, along with 11 ε-greedy trajectories in which some actions were randomly selected from the valid action set.

The early, middle, and late latent movement patterns of each trajectory were subsequently compared, and trajectories with similar movement patterns were grouped into the same branch.

### Trajectory Generation and Branch Selection

| Step | Method | Objective |
|---|---|---|
| Independent policy training | Four DQN models trained with different random seeds | Capture policy differences arising from initialization and training experience |
| Trajectory generation per model | One greedy trajectory and 11 ε-greedy trajectories | Compare representative policy behavior with exploratory variation |
| Exploration intensity | ε values ranging from 0.05 to 0.30 for non-greedy trajectories | Introduce different levels of action diversity |
| Trajectory summarization | Divide each trajectory into three segments according to relative progression | Distinguish early, middle, and late movement patterns |
| Feature representation | A 48-dimensional signature formed by concatenating standardized net displacements across 16 axes for each segment | Compare trajectories of different lengths using a consistent representation |
| Clustering | Ward hierarchical clustering | Hierarchically group trajectories with similar movement patterns |
| Number of branches | Apply an elbow criterion to the silhouette curve for $k=2$–8 | Summarize the branching structure while avoiding excessive subdivision |
| Representative trajectory | Medoid, defined as the actual simulated trajectory with the smallest total distance to other trajectories within its cluster | Present an actual generated trajectory rather than an artificially averaged path |
| Branch numbering | Order branches by the number of included simulations | Display the most frequently observed trajectory group first |

The trajectory signature is defined as:

$$
s_i=
\left[
\frac{z_{i,t_1}-z_{i,t_0}}{\boldsymbol{\sigma}},
\frac{z_{i,t_2}-z_{i,t_1}}{\boldsymbol{\sigma}},
\frac{z_{i,t_3}-z_{i,t_2}}{\boldsymbol{\sigma}}
\right]
$$

This representation compares axis-specific net displacements across three segments rather than the complete sequence of individual actions. Therefore, different detours occurring within the same segment may be summarized by an identical net displacement.

The silhouette coefficient compares the average distance within the same cluster, $a_i$, with the average distance to the nearest alternative cluster, $b_i$:

$$
s(i)=
\frac{b_i-a_i}{\max(a_i,b_i)}
$$

Higher values indicate that trajectories are more closely grouped within their own cluster and more separated from other clusters in the feature space. Rather than simply selecting the number of clusters with the highest silhouette score, the current implementation uses an elbow criterion to identify the point immediately before a substantial decline in silhouette or where further improvements become marginal.

### Figure Composition

https://github.com/user-attachments/assets/224d8fb8-943e-4238-b9e7-5d4fe73d40c7

| Panel | Display | Interpretation |
|---|---|---|
| Upper left | All trajectories as thin lines and medoids as thick lines | Diversity of generated trajectories and representative movement patterns |
| Upper right | Post-movement visitation density across all trajectories | Regions repeatedly explored during simulation |
| Lower left | Ward dendrogram | Hierarchical relationships among trajectory groups |
| Lower center | Silhouette scores across different numbers of branches | Clustering quality and the selected number of branches |
| Lower right | FOXP3 expression and entropy relative to their starting values | Expression reduction and accompanying changes in the expression distribution |

Visitation density is calculated by projecting all visited coordinates, excluding the initial step 0, onto the original PCA space. A two-dimensional histogram is then smoothed using Gaussian filtering and normalized by its maximum value. Therefore, this metric represents relative simulation visitation density rather than actual cell density or cell-fate probability. Since repeated visits are included and longer trajectories contribute more coordinates, the resulting density reflects both visitation frequency and the influence of repeated occupancy.

### Results from the Current Simulation Output

| Metric | Branch 1 | Branch 2 |
|---|---:|---:|
| Number of trajectories | 36 (75%) | 12 (25%) |
| Number of movements in representative trajectory | 500 | 500 |
| Representative trajectory termination reason | Maximum number of movements | Maximum number of movements |
| Initial FOXP3 | 0.559536 | 0.559536 |
| Final FOXP3 | 0.000322 | 0.055076 |
| Reduction relative to initial FOXP3 | ~99.94% | ~90.16% |
| Final entropy | 3.461650 | 3.504604 |
| Major net displacement axes | +Z3, +Z9, −Z14 | +Z8, +Z3, +Z10 |

Both branches showed reduced FOXP3 expression, but differed in their dominant movement axes and final reduction levels. However, the branch proportions represent the distribution of trajectories generated under the specified policies, random seeds, and exploration settings. They should not be interpreted as probabilities of actual cells adopting the corresponding biological fates.

---

## 4-3. Gaussian Virtual Cell Generation Based on Visitation Density and Expression Similarity Analysis

<img width="3510" height="2058" alt="4-2" src="https://github.com/user-attachments/assets/3fddb44a-f7ab-44ff-955a-c1220c33fe15" />

New virtual cells were generated around regions repeatedly visited by the DQN, and their expression patterns were compared across regions with different relative visitation densities. Rather than analyzing only one representative trajectory, this procedure was designed to investigate local expression diversity and consistency around the explored trajectories.

### Gaussian Sampling Strategy

All post-movement latent coordinates from the simulated trajectories were used as candidate Gaussian centers. One center was randomly selected, and small perturbations proportional to the observed standard deviation of each latent dimension were added.

$$
z_{\mathrm{virtual}} = z_{\mathrm{visit}} + \epsilon
$$

$$
\epsilon \sim N(0, \mathrm{diag}(0.03^2\sigma_1^2, \ldots, 0.03^2\sigma_{16}^2))
$$

| Step | Implementation | Rationale |
|---|---|---|
| Center coordinates | All visited points from DQN trajectories, excluding step 0 | Generate virtual cells around regions actually explored during simulation |
| Center selection | Random selection from the complete visitation list, including repeated visits | Allow exploration frequency to influence sampling proposals |
| Gaussian noise | 0.03 times the observed standard deviation of each latent dimension | Generate small latent variations around existing trajectories |
| Density evaluation | Project onto the original PCA space and interpolate the previously calculated visitation-density grid | Classify density regions using the same spatial reference |
| Candidate acceptance | Continue sampling until the target number of cells is reached for each specified density interval | Obtain sufficient samples for comparisons across density groups |
| Expression generation | Use the fixed decoder trained on 2,000 genes | Generate expression through the model rather than copying observed cell expression |
| Negative value handling | Clip only negative decoder outputs to zero | Obtain non-negative expression vectors for similarity calculations |
| Comparison target | Compare generated virtual cells with one another | Evaluate model-generated expression patterns across density groups |

### Sample Allocation Across Density Intervals

| Relative visitation density | Color | Prespecified allocation | Number of virtual cells |
|---|---|---:|---:|
| [0.8, 1.0] | Red | 68.3% | 68 |
| [0.6, 0.8) | Orange | 27.1% | 27 |
| [0.4, 0.6) | Green | 4.6% | 5 |
| Total | — | 100% | 100 |

The density values represent relative visitation density normalized by the maximum observed visitation density, not percentile-based intervals. In addition, the allocation proportions of 68.3%, 27.1%, and 4.6% were predefined for this analysis. They do not represent the actual probability mass of each region or the 1σ, 2σ, and 3σ intervals of a Gaussian distribution.

Gaussian samples are generated independently of DQN actions. Therefore, they are not automatically guaranteed to satisfy the original trajectory constraints, including monotonic FOXP3 reduction or the empirical cell-support condition.

### Expression Similarity Calculation

Cosine similarity was calculated between non-negative decoder output vectors, x and y, representing virtual cells:

$$
\mathrm{cosine}(x,y) =
\frac{x^T y}
{\sqrt{\sum_i x_i^2}\sqrt{\sum_i y_i^2}}
$$

Values approaching 1 indicate that the overall gene expression vectors have similar directions. The calculation uses log1p decoder outputs with only negative values clipped to zero; the values are not inverse-transformed into linear counts.

| Comparison | Matrix size | Objective |
|---|---:|---|
| Mean expression vectors across density groups | 3 × 3 | Compare overall mean expression patterns between density intervals |
| All virtual cell pairs | 100 × 100 | Evaluate expression similarity across the entire sample |
| Within-group cell pairs | 68 × 68, 27 × 27, 5 × 5 | Assess within-group diversity |
| Between-group cell pairs | Pairwise combinations across density groups | Compare expression differences and distributions between groups |

### Between-Group Similarity Results from the Current Saved Output

| Density group comparison | Cosine similarity between mean vectors | Median cosine similarity between individual cell pairs |
|---|---:|---:|
| High–Intermediate density | 0.999999 | 0.999972 |
| High–Low density | 0.999976 | 0.999934 |
| Intermediate–Low density | 0.999986 | 0.999919 |

All three comparisons showed cosine similarities very close to 1, indicating that the overall expression patterns of the generated virtual cells remained highly similar across visitation-density groups. Among individual cell-pair comparisons, high–intermediate density showed the highest median similarity, whereas intermediate–low density showed the lowest. However, the differences were small, on the order of approximately 10^-5. Therefore, they are more appropriately interpreted as subtle local expression differences rather than clear separation between distinct cellular states.

Similarity between group-level mean vectors was higher than similarity between individual cell pairs, potentially because averaging reduces variation among individual samples. The heatmap also appears nearly uniform because all values are close to 1, whereas the boxplot provides a magnified view of these small differences.

---
[Click here to view the Reverse scDQ-path Virtual Cell Sampling notebook](https://github.com/AiJayce/scDQ-path/blob/Virtual-Trajectory-Construction/scDQN.ipynb)
---
