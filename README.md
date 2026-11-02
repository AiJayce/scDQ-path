# Advanced : Double KO Simulation

## 6-1. Simulation of a Joint ATF6–IRE1 Suppression Trajectory

Extending the single-gene suppression approach, a virtual Double KO trajectory was explored to simultaneously reduce **ATF6 and IRE1**. Since the official gene symbol for IRE1 is **ERN1**, the decoder outputs of ATF6 and ERN1 were used as the two target genes in the model.

The expression space was learned using real control and ATF6–IRE1 double-perturbation cells. A DQN was then trained to jointly reduce the expression of both targets from a representative control coordinate. No specific destination coordinate was predefined.

### Double KO Model Configuration

<img width="3709" height="2211" alt="6-1" src="https://github.com/user-attachments/assets/09d57131-9147-4ca3-86fe-6569d0fa80fc" />

| Component | Implementation | Purpose |
|---|---|---|
| Training conditions | `3x_neg_ctrl_pMJ144-2` and `ATF6_IRE1_pMJ152` | Learn the expression structure of control and real double-perturbation cells |
| Representation space | VAE trained on 3,000 genes with a 128-dimensional latent space | Compress high-dimensional expression profiles for latent-space exploration |
| Starting point | Virtual coordinate obtained by inverse PCA transformation of the dimension-wise median control PCA coordinates | Standardize the initial state across all trajectories |
| Actions | Movement along positive and negative latent axes by 0.01 times the observed standard deviation | Generate trajectories through repeated small state transitions |
| Action constraints | Both ATF6 and ERN1 must decrease, and the resulting expression values must remain non-negative | Prevent a decrease in one target from compensating for an increase in the other |
| Reward | Mean of the two target reductions, normalized by their respective starting expression levels | Jointly evaluate targets with different expression magnitudes |
| Support constraint | Distance threshold based on the fifth-nearest real cell | Restrict excessive deviation from the observed distribution |
| Trajectory generation | 48 simulations across four independent DQN models | Explore diverse joint-suppression trajectories |
| Representative trajectories | Branching using Ward clustering and the silhouette elbow method, followed by medoid selection | Summarize similar trajectories into representative branches |
| Evaluation expression | Gaussian sampling around trajectory coordinates, followed by reconstruction using the fixed decoder | Generate virtual cell populations for comparison with real perturbation cells |

The reward was defined as follows:

$$
r_t = 0.5 \left[
\frac{ATF6_t-ATF6_{t+1}}{ATF6_{\mathrm{start}}}
+
\frac{ERN1_t-ERN1_{t+1}}{ERN1_{\mathrm{start}}}
\right]
$$

Only actions that decreased both targets were permitted. Entropy was used exclusively as a monitoring metric. Trajectories were terminated after a maximum of 500 movements, when no valid actions remained, or when the net displacement became stagnant during simulation.

https://github.com/user-attachments/assets/437f6df4-a4dd-4e61-8d67-43de5d9d2e75

In the attached execution, the 48 simulations were separated into **five branches**. This indicates that distinct movement patterns can emerge even when both targets are jointly suppressed. However, **joint suppression does not necessarily imply complete elimination of both target expressions**, and the Gaussian samples used for evaluation were not independently constrained to follow a monotonic decrease.

---

## 6-2. Comparison with Real Double and Single Perturbations: Unexpected Expression Similarity

To assess whether the generated Double KO population reproduced the characteristics of real double-perturbation cells, the **same virtual Double KO expression population** was independently compared with the following three real conditions:

- ATF6–IRE1 Double perturbation: `ATF6_IRE1_pMJ152`
- ATF6 Single perturbation: `ATF6_only_pMJ145`
- IRE1 Single perturbation: `IRE1_only_pMJ148`

All comparisons used the same 2,000 evaluation genes and expression scale. Single-perturbation cells were excluded from representation learning and trajectory generation and were used only during the comparison stage.

### Comparison Results

<img width="3870" height="1951" alt="6-2" src="https://github.com/user-attachments/assets/623d5bb8-4cef-4011-8794-d4d0584f49a4" />

| Real Comparison Condition | Cosine Median ↑ | MAE ↓ | RMSE ↓ |
|---|---:|---:|---:|
| ATF6–IRE1 Double KO | 0.8780 | 0.002009 | 0.018062 |
| ATF6 Single KO | 0.9013 | 0.001938 | 0.017647 |
| **IRE1 Single KO** | **0.9451** | **0.001443** | **0.008464** |

The virtual Double KO showed **higher expression similarity to the IRE1 Single KO condition than to the real Double KO condition**. Both MAE and RMSE of gene-wise mean expression were also lower in the IRE1 Single KO comparison.

Despite being designed to suppress both targets simultaneously, the generated population was more similar to the single IRE1 perturbation condition. This raised the question of whether **the current simulation sufficiently reproduces the transcriptional response specific to Double KO**. Satisfying the suppression constraints for both targets alone does not establish that the overall expression state of a double perturbation has been reproduced.

Accuracy was not assessed solely from the central position of the residual boxplot. MAE and RMSE were also examined because positive and negative residuals can cancel each other, resulting in a small median or mean even when substantial errors remain for individual genes.

Furthermore, real Double KO cells were included in DEG selection, VAE training, and support construction. Therefore, this comparison does not constitute an independent holdout validation. The results were interpreted as an **exploratory validation of condition-specific expression similarity**.

---

## 6-3. Examination of Target Expression Differences and Hypotheses for the Results

<img width="3952" height="919" alt="6-3" src="https://github.com/user-attachments/assets/d28cfbdd-d754-4bb4-8999-acd6899a47eb" />

To investigate possible explanations for the unexpected comparison results, the **measured expression distributions and mean expression levels of ATF6 and ERN1** were examined between the control and real Double KO conditions.

In the figure, ATF6 showed a relatively high mean expression level in the control condition and a substantial decrease in the Double KO condition. In contrast, ERN1 already exhibited low expression in the control condition, resulting in a **smaller absolute difference in mean expression between the two conditions**.

| Observation | ATF6 | IRE1 / ERN1 |
|---|---|---|
| Mean expression in control | Relatively high | Relatively low |
| Absolute mean difference from Double KO | Large | Relatively small |
| Consideration for interpretation | Target suppression signal is relatively distinct | Low expression and sparsity may make the suppression signal more difficult to distinguish |

Based on these observations, the possibility that **differences in the magnitude and detectability of the observed suppression signals between the two targets influenced the simulation results** was considered. In particular, for a target such as ERN1 with low baseline expression, limited observed expression signals may affect representation learning and the estimation of decoder responses.

However, a small **absolute difference in mean expression** does not necessarily indicate a low suppression rate or a statistically insignificant difference. When baseline expression is low, the relative reduction can still be substantial despite a small absolute change.

Furthermore, this figure compares only two target genes between the control and Double KO conditions. Therefore, it cannot establish why the virtual Double KO showed greater similarity to the IRE1 Single KO condition. This explanation remains a hypothesis requiring further investigation through relative expression reduction, detection rate, and downstream response gene analyses, including the single-perturbation conditions.

---

## Conclusion

This analysis demonstrated that **generating a trajectory that reduces target expression and reproducing the actual transcriptomic response of a KO or Double KO are distinct objectives**.

Therefore, before performing KO simulations, it is necessary to determine whether suppression of each target is sufficiently observable in the real data. In particular, for Double KO simulations, the following aspects should be examined for both targets:

- **Absolute and relative expression changes** compared with the control condition
- Baseline expression levels and target detection rates
- **Statistical evidence for expression changes** based on an appropriate replicate and testing design
- Consistent responses of downstream genes and associated pathways

**In other words, rather than judging simulation success solely by high overall expression similarity, it is essential to first establish whether the underlying suppression signals are sufficiently strong and reproducible.** The current mean expression plots provide exploratory evidence motivating this requirement, but do not establish statistical significance.

----
[Click here to view the Double KO Simulation notebook](https://github.com/AiJayce/scDQ-path/blob/Advanced-Double-KO-Simulation/Double-KO.ipynb)
----


