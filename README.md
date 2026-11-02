# 2. Virtual Trajectory Construction

# 2-1. DQN-Based Virtual Cellular State Trajectory Exploration in the VAE Latent Space

<img width="3612" height="1671" alt="2-1" src="https://github.com/user-attachments/assets/169ea3b4-5591-4dfc-a6b8-6cdf17e8512b" />

A virtual trajectory was explored in the latent space learned from single-cell gene expression data, with the objective of transitioning from a representative CD4 Naive T-cell state toward a representative Regulatory T-cell state. First, highly variable genes were selected from the input cells while accounting for sample composition. The two target cell populations were then extracted and used to train a VAE. This enabled high-dimensional gene expression profiles to be compressed into a low-dimensional latent representation, where a DQN was configured to select both movement directions and step lengths.

Unlike approaches that construct trajectories by sequentially connecting observed cells, this method generates new virtual coordinates by adding an action vector to the current latent coordinate. Each generated coordinate is subsequently passed through the trained VAE decoder to reconstruct gene expression profiles, enabling the exploration of transcriptional changes along the virtual trajectory toward the target state.

## Algorithm Configuration

| Component | Implementation | Rationale |
|---|---|---|
| Target populations | CD4 Naive T and Regulatory T cells | To focus on the cellular state differences between the two populations. |
| Input expression | Normalized and log-transformed expression of 2,000 highly variable genes | To retain major cellular variation while reducing input dimensionality and noise. |
| Representation learning | β-VAE using expression reconstruction error and KL divergence | To learn a latent space capable of reconstructing gene expression profiles. |
| Latent coordinates | 16-dimensional posterior mean estimated by the encoder | To minimize coordinate variability caused by stochastic sampling during repeated encoding. |
| Initial state | Virtual latent coordinate obtained by inverse PCA transformation of the coordinate-wise median of Naive T cells | To define a representative starting point independent of any individual observed cell. |
| Target state | Virtual latent coordinate obtained by inverse PCA transformation of the coordinate-wise median of Regulatory T cells | To establish a representative destination for the exploration process. |
| Environment state | Continuous 16-dimensional latent coordinates generated through actions | To allow exploration of intermediate coordinates that are not necessarily represented by observed cells. |
| DQN input | 37 features, including current coordinates, displacement and distance to the target, current and cumulative minimum entropy, current step, and remaining movement budget | To incorporate both the current position and the relationship to the target under a limited movement budget. |
| Movement directions | Target direction, opposite target direction, and positive/negative directions along all 16 latent dimensions | To allow both direct target-oriented movement and exploration of surrounding directions. |
| Movement lengths | Maximum step length (L) and half-step length (0.5L) for each direction | To enable both larger movements and fine-grained exploration. |
| Total action space | 34 directions × 2 movement lengths = **68 discrete actions** | To represent continuous latent-space movement as a finite action space that can be optimized by DQN. |
| State transition | $z_{t+1}=z_t+\Delta z(a_t)$ | To directly generate the next virtual state using the selected action. |
| Movement limit | $L=2D/500$, where D is the initial distance between the starting and target coordinates | To construct trajectories through accumulated small movements rather than large jumps. |
| Action constraints | Action masking based on empirical cellular support and the feasibility of reaching the target within the remaining steps | To restrict movements that excessively deviate from the observed data distribution or make target arrival infeasible within the remaining budget. |
| Exploration strategy | ε-greedy during training and maximum-Q action selection among valid actions during evaluation | To encourage exploration during training and evaluate the learned policy under constrained action selection. |
| Value learning | Experience replay and target networks, with Double DQN separating next-action selection from value estimation | To improve training stability by reducing temporal correlation between experiences and overestimation of action values. |
| Policy selection | Selection of the checkpoint with the highest discounted cumulative reward among evaluation trajectories | To evaluate overall trajectory performance rather than relying on individual step rewards. |
| Termination conditions | Target arrival, convergence in distance improvement, or a maximum of 500 actions, whichever occurs first | To prevent unnecessary movements and repeated exploration under stagnation. |
| Expression reconstruction | Input of each virtual latent coordinate into the trained VAE decoder | To connect latent-space transitions with corresponding gene expression changes. |

The support constraint is evaluated using the distance to the **fifth nearest observed cell** from each candidate destination. However, in the current implementation, target-direction actions are exempted from this constraint. Therefore, the resulting trajectory is not guaranteed to remain entirely within high-density regions of the observed cellular distribution.

## Reward Design

Let $d_t$ denote the distance to the target at step $t$, and let $\ell_t$ denote the actual movement length. The reward function is defined as follows:

$$
r_t = \frac{d_t-d_{t+1}}{L} + \min(1,\max(0,1-\frac{\ell_t}{L})) - 0.03 + \mathbb{1}(\text{first target arrival})
$$


| Reward Component | Calculation and Interpretation | Objective |
|---|---|---|
| Target approach reward | Positive when moving closer to the target and negative when moving farther away | To encourage movement toward the representative Regulatory T-cell coordinate. |
| Short movement reward | 0 when movement length equals L and 0.5 when movement length equals 0.5L | To assign additional value to shorter movements rather than favoring large jumps. |
| Step cost | A constant penalty applied to every action | To discourage unnecessary increases in the number of movements. |
| Target arrival bonus | Awarded only upon the first target arrival, followed by immediate termination | To provide additional incentive for reaching the target while preventing repeated arrival bonuses. |
| Training reward scaling | The combined reward is multiplied by 0.1 during training | To adjust the numerical scale of rewards for Q-learning. |
| Long-term reward | A discount factor of 0.99 is applied to future rewards | To account for future trajectory outcomes in addition to immediate movement benefits. |

The target approach and short movement components have equal weights. Consequently, moving a full distance of L toward the target and moving half that distance can produce the same combined immediate score from these two components. **The short movement reward does not inherently guarantee the shortest trajectory or directional smoothness.**

Additionally, decoder-based Shannon entropy is used to characterize the reconstructed expression profiles and monitor trajectory-associated changes. However, **entropy is not included in the current reward function.**

***Notice***

<img width="4479" height="1834" alt="2-2" src="https://github.com/user-attachments/assets/3ba16021-fc1a-4920-9e50-87e40e1234b4" />

>> 1. The DQ-path algorithm may not perform reliably when cellular clusters are poorly separated. Therefore, after Leiden-based clustering using a neighbor graph, it is recommended to remove subclusters that exhibit insufficient separation from neighboring clusters before applying DQ-path.

>> 2. The tolerance threshold can be adjusted between 40 and 60 to filter out dropout-associated subclusters.
>> 3. The Silhouette Coefficient can be used as a quantitative criterion to assess clustering quality before applying DQ-path.
>> [Click here](https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt) to download the TF list. If HVG filtering excludes TFs of interest, it is recommended to construct the UMAP using HVGs first and then add the desired TFs to the selected gene list for downstream analysis.

### Silhouette Score Evaluation

| Silhouette Score | Clustering Quality | Recommendation |
|---|---|---|
| 0.71–1.00 | Strong separation | Highly suitable |
| 0.51–0.70 | Reasonable separation | Suitable |
| 0.26–0.50 | Weak separation | Consider refining clustering |
| ≤ 0.25 | Poor separation | Reconsider clustering or remove poorly separated subclusters |
| Negative | Potentially incorrect cluster assignment | Review cluster composition and assignments |
---

# 2-2. Action-Level Trajectory Validation and Policy Diagnostics

<img width="4397" height="1350" alt="2-3" src="https://github.com/user-attachments/assets/21b1a374-7bd9-43d5-a975-3d9b81a69c0f" />

To evaluate trajectories generated by DQN, the latent coordinates before and after each action, target distance, movement length, entropy, reward, and selected action Q-value were recorded.

Rather than evaluating only whether the final coordinate approaches the target, this procedure examines **whether each movement satisfies the defined constraints, how the movement is rewarded, and whether the recorded termination condition is consistent with the actual trajectory.**

Action-level validation does not involve repeatedly retraining the model using an independent validation dataset. Instead, it represents a trajectory auditing procedure that combines movement constraint checks, independent recalculation of outcomes, and policy diagnostics.

## Action-Level Validation Criteria

| Validation Component | Per-Action Inspection | Purpose |
|---|---|---|
| Action validity | Verification that the selected action passes the current action mask | To ensure that prohibited actions are not selected by the policy. |
| State transition consistency | Comparison between the recorded candidate displacement and the displacement recalculated from the current coordinate | To verify that a movement calculated from an incorrect state is not applied. |
| Actual movement length | Calculation of the Euclidean distance between consecutive coordinates and verification of the L upper bound | To confirm that the intended movement constraint is maintained in the actual latent space. |
| Empirical cellular support | Measurement of the distance from the candidate coordinate to its fifth nearest observed cell | To identify movements into regions excessively distant from the observed cellular distribution. |
| Remaining-step feasibility | Comparison of the candidate-to-target distance against the maximum reachable distance within the remaining action budget | To determine whether detours make target arrival infeasible within the remaining steps. |
| Target approach | Calculation of $d_t-d_{t+1}$ | To determine whether each movement actually approaches or moves away from the target. |
| Reward decomposition | Independent recalculation of target approach, short movement, step cost, and arrival bonus | To distinguish whether a high reward originates from actual target progression or another reward component. |
| Reward reproducibility | Comparison between the recalculated total reward and the recorded rollout reward | To verify consistency across training, evaluation, and trajectory storage. |
| Decoder entropy | Recording of $H_t$ and $\Delta H$ before and after each movement | To characterize changes in reconstructed expression profiles along the trajectory. |
| Candidate entropy auditing | Recording entropy, action validity, and selection status for all candidates, followed by comparison of the selected candidate with the actual destination | To verify that the entropy values used for diagnostics correspond to the actual selected movement. |
| Selected action Q-value | Recording of the predicted Q-value for the selected action | To examine how the policy's estimated long-term value changes across trajectory segments. |
| Discounted cumulative reward | Calculation of accumulated rewards until termination | To evaluate the overall trajectory according to the objective function rather than individual movements. |
| Distance convergence | Inspection of improvements in the cumulative minimum target distance over the most recent 20 movements | To identify stagnation without actual target arrival. |
| Target arrival verification | Verification of target distance in the full 16-dimensional latent space | To avoid interpreting apparent overlap in PCA visualization as actual target arrival. |
| Post-termination records | Use of trajectory coordinates only up to the first termination event and verification of the expected number of coordinates (number of actions + 1) | To prevent post-termination movements or artificially connected target coordinates from being included. |
| Visualization consistency | Verification that arrow displacements and colors correspond to actual coordinate changes and $\Delta H$ | To ensure that the visualization accurately represents the recorded trajectory. |

Q-values and entropy curves are **diagnostic metrics** and do not independently establish policy accuracy or biological validity. In particular, Q-values estimate expected future discounted rewards and are not necessarily expected to increase continuously along the trajectory.

## Visualization Interpretation

In the left PCA panel, background points represent observed cells, while the connected trajectory represents movements between virtual latent coordinates generated by DQN. Trajectory colors indicate entropy changes between consecutive actions: blue represents a decrease, whereas red represents an increase. The pink diamond marks the coordinate with the lowest entropy among the generated trajectory points, excluding the initial state. It does not represent the global minimum across the entire latent space.

The right diagnostic panel jointly displays entropy changes, discounted cumulative rewards, selected action Q-values, distance to the target, and individual reward components. By examining these metrics together, it becomes possible to distinguish **whether increasing rewards correspond to actual target progression** and whether repetitive movements or stagnation occur near the target.

In the illustrated execution, the trajectory terminated at action 283 due to the `distance_converged` condition. This indicates that improvements in target distance had converged, **not that the target had been reached exactly**. Therefore, the actual termination coordinate was displayed as a black square to distinguish it from the target coordinate.

> When regenerating the figure using the current implementation, the entropy-related legend in the reward panel should be renamed to **Short Latent Movement Reward**. The termination step and the subsequent TF analysis should also be synchronized with the same execution result.

---

# 2-3. Analysis of Transcription Factor Expression Changes Along the Virtual Trajectory

<img width="4313" height="1353" alt="2-4" src="https://github.com/user-attachments/assets/b944c164-bd2f-4572-9b93-6fecfca70cfd" />

To investigate molecular changes along the DQN-generated trajectory, each virtual latent coordinate was passed through the VAE decoder to reconstruct its corresponding gene expression profile.

The intersection between a reference transcription factor (TF) list and the genes available in the decoder output was used to identify candidate TFs. The top 10 TFs with the highest sample variance in reconstructed expression across the entire trajectory were subsequently selected.

Variance-based selection captures not only differences between the initial and terminal states but also expression fluctuations occurring throughout intermediate trajectory segments. However, **high variance indicates expression variability along the generated trajectory and does not directly establish regulatory effects or causal importance.**

| Visualization | Displayed Information | Analytical Objective |
|---|---|---|
| Left bar plot | Top 10 TFs ranked by variance in reconstructed decoder expression across the trajectory | To identify TFs exhibiting substantial expression variability along the virtual trajectory. |
| Upper-right annotation strip | Proportion of Regulatory T cells among the 10 nearest observed cells at each trajectory coordinate | To examine whether the local cellular composition shifts from Naive T-cell-dominated regions toward Regulatory T-cell-enriched regions. |
| Lower-right heatmap | Step-wise reconstructed expression of the same top 10 TFs | To characterize TF-specific expression patterns and identify trajectory segments associated with expression changes. |

The upper annotation strip represents the **fraction of Regulatory T cells among the 10 nearest observed cells**, rather than a distance-weighted score or classification probability.

TF expression values are directly reconstructed by the VAE decoder and are not calculated as averages of neighboring observed cells.

For heatmap visualization, each gene's expression values were independently min-max scaled across the trajectory, with the minimum mapped to 0% and the maximum mapped to 100%. Therefore, the colors represent **relative expression changes within each individual gene**, rather than absolute expression differences between different genes.

In the illustrated results, reconstructed expression levels of BACH2, LEF1, and MYC decreased toward the later trajectory segments, whereas FOXP3 and IKZF2 exhibited increasing expression patterns.

Concurrently, the proportion of neighboring observed Regulatory T cells increased along the trajectory. These patterns allow us to investigate whether virtual cellular state transitions are accompanied by coordinated changes in reconstructed TF expression and the local composition of observed cells.

----
[Click here to view the Virtual Trajectory Construction notebook](https://github.com/AiJayce/scDQ-path/blob/Virtual-Trajectory-Construction/scDQN.ipynb)
----
