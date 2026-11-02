# Gene-Regulatory-Network (GRN) Construction

## 3-1. Virtual Gene Expression Response Analysis Following FOXP3 Suppression

<img width="3203" height="2258" alt="3-1" src="https://github.com/user-attachments/assets/15f4e036-fcf8-46f0-9695-1e81f4b2937d" />

The Treg-like region of the DQN-generated trajectory was selected to investigate how the expression of other genes changes when FOXP3 expression is suppressed toward zero. Instead of performing actual gene editing, a counterfactual simulation was conducted by **optimizing latent coordinates while keeping the trained VAE decoder fixed**.

Let $z_s$ denote the baseline latent coordinate, $z'_s$ the optimized latent coordinate, and $f$ the decoder. The following objective function was minimized:

$$
\mathcal{L} = \rho \frac{1}{|S|}\sum_{s\in S}f_{\mathrm{FOXP3}}(z'_s)^2 + 0.1\frac{1}{|S|}\sum_{s\in S}(z'_s-z_s)^2 + 0.001\frac{1}{|S|}\sum_{s\in S}(z'_s)^2
$$

| Objective Component | Description | Purpose |
|---|---|---|
| FOXP3 output suppression | Minimizes the decoder output of FOXP3 toward zero | To generate a FOXP3-zero condition |
| Latent coordinate proximity | Penalizes deviations from the original latent coordinates | To limit excessive state changes unrelated to FOXP3 suppression |
| Latent coordinate regularization | Penalizes the magnitude of optimized latent coordinates | To prevent extreme latent-space displacement |

The FOXP3 suppression penalty ($\rho$) was progressively increased during optimization. The resulting coordinates were evaluated to confirm that the absolute FOXP3 decoder output was within the specified tolerance.

This procedure does not artificially overwrite the decoded FOXP3 expression with zero after reconstruction. Instead, it **searches for latent coordinates at which the decoder itself produces FOXP3 expression close to zero**.

For each gene $g$, the response was calculated as the paired difference between the KO and baseline conditions at the same trajectory coordinates, followed by median aggregation:

$$
\Delta_g = \mathrm{median}_{s\in S}\left[f_g(z'_s)-f_g(z_s)\right]
$$

The histogram shows the distribution of $\Delta_g$ across genes, excluding FOXP3 itself. The 90th percentile of absolute expression changes was used as an empirical threshold to identify relatively large responses. Boxplots were also generated to compare the baseline and KO expression distributions of selected marker genes.

In the presented results, CTLA4 and IKZF2 showed decreased expression, whereas CCR7 showed an increase. These changes represent model-predicted expression responses associated with FOXP3 suppression and **do not establish causal effects observed in an actual KO experiment**.

> The asterisks and significance labels in the figure indicate an empirical threshold corresponding to approximately the top 10% of absolute expression changes. They do not represent statistical significance based on p-values or FDR.

---

## 3-2. Directional and Signed TF KO Response Network

<img width="4267" height="2129" alt="3-2" src="https://github.com/user-attachments/assets/bc4b3108-9d25-43bd-9db1-26ef4d6f8ea5" />

Independent virtual KO simulations were performed for the top 10 transcription factors (TFs) exhibiting the highest expression variability along the DQN-generated trajectory. For each TF, the reconstructed expression responses of other genes following its suppression were calculated within the Treg-like trajectory region. These relationships were then assembled into a **directional network in the form of TF → response gene**.

This network does not represent direct DNA-binding relationships. Instead, it describes **model-predicted expression responses associated with the virtual KO of individual TFs**.

### Network Power Definition

For the virtual KO of TF $i$, the response strength of gene $j$ was defined as follows:

$$
P_{ij} = \mathrm{median}_{s\in S}\left[f_j(z_s^{KO(i)})-f_j(z_s)\right]
$$

- $S$: Set of Treg-like DQN trajectory coordinates used for analysis
- $z_s^{KO(i)}$: Optimized coordinate at which the decoder output of TF $i$ is suppressed toward zero
- $f_j$: Expression of gene $j$ reconstructed by the fixed decoder
- $P_{ij}$: Signed network power

Network power is calculated as the **median of paired differences across corresponding coordinates**. It is not equivalent to subtracting the separately calculated medians of the two conditions.

Furthermore, power represents a difference in the log1p CP10K expression scale. It is not a correlation coefficient, probability, or fold change.

For each TF, the absolute values of $P_{ij}$ were calculated for all genes except the TF itself. Only nonzero relationships at or above the 90th percentile of the corresponding TF-specific absolute power distribution were selected as edges.

Consequently, the edge-selection threshold differs between TFs. The resulting network does not represent relationships selected using a common statistical significance threshold.

### Visual Representation of Nodes and Edges

| Visual Element | Meaning | Interpretation |
|---|---|---|
| Orange triangle | TF used as the KO source | Indicates outgoing response relationships originating from the KO-targeted TF |
| Purple triangle | Other TF nodes | Can act as response targets of other TF KOs |
| Light gray circle | Non-TF genes | Represents selected KO-responsive target genes |
| Inner TF layer | Region containing TF nodes | A layout for visualization, not a biological location or distance |
| Outer target layer | Region containing general target genes | A layout used to improve network readability |
| Arrow ($i \rightarrow j$) | Expression response of gene $j$ following virtual KO of TF $i$ | Does not establish direct binding or temporal signal transmission |
| Blue edge | $P_{ij}<0$: target expression decreases after KO | **Decreased on KO** |
| Red edge | $P_{ij}>0$: target expression increases after KO | **Increased on KO** |
| Light-colored edge | Relatively small expression response | Color represents a continuous scale across the network-wide power range |
| Thick edge | Large $|P_{ij}|$ | Represents a larger predicted response, not greater statistical confidence |
| Node size | Distinction between node types | Does not represent centrality or expression level |

Edge thickness was normalized using the maximum absolute network power:

$$
w_{ij} = 0.4 + 2.6\frac{|P_{ij}|}{\max_{(u,v)\in E}|P_{uv}|}
$$

Therefore, thicker edges represent larger predicted expression responses, but do not indicate greater statistical reliability.

### Relationship Between Up/Down Regulation and Edge Colors

The edge colors represent the **direction of target expression changes following KO**, rather than the regulatory sign of the normal TF.

| Target Expression After KO | Edge Color | Limited Interpretation of Normal TF Function |
|---|---|---|
| Decreased target expression | Blue | Consistent with a potential contribution of the TF to maintaining target expression |
| Increased target expression | Red | Consistent with a potential inhibitory effect of the TF on target expression |

For example, if a gene decreases following FOXP3 KO, the corresponding edge is displayed in **blue**. This pattern is compatible with the possibility that normal FOXP3 activity supports the expression of that gene. However, it does not establish a direct activation relationship. Indirect responses and the learned structure of the decoder may also contribute to the observed pattern.

### Total Network and Local Networks

The **Total Network** combines the selected response relationships from successfully optimized virtual KOs among the top 10 TFs, provided that each KO satisfies the specified tolerance criterion. In the presented execution, the network contained 345 nodes and 1,200 edges.

The **Local Networks** separately display outgoing response relationships for BACH2, FOXP3, and PRDM1, which ranked among the top three TFs by expression variance along the trajectory.

These TFs were selected based on **expression variability, not network centrality**.

The Local Networks use the same coordinate, color, and edge-width scales as the Total Network, allowing comparison of TF-specific response targets and predicted response magnitudes.

---

## 3-3. Centrality Analysis of the Global TF KO Response Network

<img width="4343" height="2031" alt="3-3" src="https://github.com/user-attachments/assets/0ef782e0-8b10-4e82-9720-90169983dfb0" />

To extend the analysis beyond TFs with high expression variability, virtual KO simulations were attempted for all TFs that could be reconstructed by the decoder. In the presented execution, 141 TFs were included as candidates.

A **global directed, unweighted graph** was constructed from the selected TF–target relationships obtained from successful KO simulations. Centrality metrics were calculated using the entire graph, including both TFs and non-TF genes. The top 10 TFs were then extracted for each centrality metric.

Therefore, the reported results were not calculated using a TF-only subgraph.

For this centrality analysis, the sign and magnitude of network power were not considered. Instead, calculations were based solely on the **presence and direction of selected connections**.

### Centrality Metrics and Definitions

Let $N$ denote the total number of nodes in the graph, and let $A_{uv}=1$ indicate the presence of a directed edge from node $u$ to node $v$.

#### 1. Betweenness Centrality: Mediation of Shortest Paths

$$
C_B(v) = \frac{1}{(N-1)(N-2)}\sum_{s\ne v}\sum_{\substack{t\ne v\\t\ne s}}\frac{\sigma_{st}(v)}{\sigma_{st}}
$$

- $\sigma_{st}$: Number of directed shortest paths from node $s$ to node $t`
- $\sigma_{st}(v)$: Number of those paths passing through node $v$

A high value indicates that a node frequently appears along the shortest paths connecting other nodes.

In this network, a high betweenness value identifies TFs occupying **intermediary positions connecting different KO-response relationships**.

In the presented results, CERS6 exhibited the highest betweenness centrality.

However, these graph paths represent computationally connected response relationships and do not imply that molecular signals are transmitted in the same sequence in actual cells.

#### 2. Incoming Closeness Centrality: Accessibility from Other Nodes

$$
C_C^{in}(v) = \frac{r_v-1}{N-1}\cdot\frac{r_v-1}{\sum_{u\in R_v,\ u\ne v}d(u,v)}
$$

- $R_v$: Set of nodes that can reach node $v$ by following edge directions, including $v$ itself
- $r_v=|R_v|$: Number of nodes in the reachable set
- $d(u,v)$: Directed shortest-path distance from node $u$ to node $v`

A high value indicates that a node can be reached from many other nodes through relatively short directed paths.

Therefore, high incoming closeness identifies TFs that are **accessible to responses originating from multiple KO sources**. It does not directly indicate the ability of a TF to influence a large number of downstream genes.

#### 3. Total Degree Centrality: Breadth of Direct Connections

$$
C_D(v) = \frac{k_v^{in}+k_v^{out}}{N-1}
$$

- $k_v^{in}$: Number of incoming connections representing responses of the TF to other TF KOs
- $k_v^{out}$: Number of outgoing connections from the TF to its responsive target genes

A high value indicates that a node participates in many direct response relationships.

In the presented results, the high degree centrality of PRDM1 and FOXP3 represents their **combined number of incoming and outgoing connections**, rather than their expression levels or regulatory strength.

In a directed graph, the sum of in-degree and out-degree can theoretically exceed 1 after normalization by $N-1$. Furthermore, outgoing connection counts are influenced by the TF-specific percentile-based edge-selection procedure. Therefore, degree alone should not be interpreted as regulatory strength.

#### 4. Incoming Eigenvector Centrality: Connections to Important Sources

$$
\lambda x_v = \sum_{u:u\rightarrow v}x_u
$$

$$
\sum_v x_v^2=1
$$

Unlike simple incoming connection counts, this metric assigns higher scores to nodes receiving connections from other nodes with high centrality scores.

Therefore, a high value indicates a response TF connected to central nodes in the network. Since this is an incoming-direction metric, it is more appropriate to interpret it as a **receiving position connected to central response sources**, rather than as strong regulatory activation of other genes.

#### 5. PageRank: Importance Accumulated Through Directed Connections

$$
PR(v) = \frac{1-\alpha}{N} + \alpha\left[\sum_{u\rightarrow v,\ k_u^{out}>0}\frac{PR(u)}{k_u^{out}}+\frac{\sum_{u:k_u^{out}=0}PR(u)}{N}\right]
$$

The damping factor was set to $\alpha=0.85$ in the current implementation. Scores from nodes without outgoing connections were redistributed uniformly across all nodes.

PageRank assigns higher scores to nodes receiving connections from important nodes, while distributing each source node's score across its outgoing connections.

Therefore, it emphasizes **target TFs accumulating importance through directed relationships in the response network**.

### Joint Interpretation of Centrality Metrics

| Metric | Meaning of a High Value | What It Does Not Indicate |
|---|---|---|
| Betweenness | Mediates multiple shortest connection paths | Confirmation of actual signal transmission order |
| Incoming closeness | Reachable through short paths from multiple sources | Broad outgoing regulatory influence |
| Total degree | Participates in many direct response relationships | Large network power or strong causal effects |
| Incoming eigenvector | Receives connections from central source nodes | Direct transcriptional activation |
| PageRank | Accumulates importance through connections from important sources | Expression level or KO effect magnitude |

In the bar plots, **orange indicates TFs with successful source KO simulations**, whereas **gray indicates TFs that did not satisfy the KO tolerance criterion**.

TFs with failed KO simulations do not generate outgoing connections, but their incoming connections from other successfully simulated TFs are retained. Consequently, gray TFs can still appear among the top-ranked nodes for incoming centrality metrics.

> KO failure or low centrality should not be interpreted as evidence of biological irrelevance. These rankings summarize the topological properties of the model-derived response network under the specified empirical edge-selection criteria.

---

## 3-4. Cell-Level State Change Simulation Using the FOXP3 KO Response Network

<img width="2593" height="1977" alt="3-4" src="https://github.com/user-attachments/assets/19950b13-359e-44ef-9d21-f60f4808a23d" />

The FOXP3 KO responses estimated along the Treg-like DQN trajectory were applied to individual observed cells to visualize the **predicted direction and magnitude of cellular state changes**.

This analysis does not independently optimize KO latent coordinates for each cell or retrain the DQN. Instead, it performs a **single-step perturbation simulation by scaling the previously estimated FOXP3 response network according to each cell's baseline FOXP3 expression**.

In the presented execution, 200 FOXP3 target relationships were applied to 3,152 cells.

### 1. Baseline Expression Reconstruction

The original latent coordinate of each observed cell ($z_c$) was passed through the fixed decoder.

$$
x_c=f(z_c)
$$

Here, $x_c$ represents the reconstructed log1p expression rather than the original observed counts. The existing VAE and PCA transformations were kept unchanged.

### 2. Cell-Specific FOXP3 Response Scaling

The median baseline FOXP3 expression across the Treg-like trajectory coordinates used for network estimation was defined as the reference value.

$$
F_{\mathrm{ref}} = \mathrm{median}_{s\in S}\left[f_{\mathrm{FOXP3}}(z_s)\right]
$$

The response scaling factor for cell $c$ was calculated as follows:

$$
a_c = \frac{\max(x_{c,\mathrm{FOXP3}},0)}{F_{\mathrm{ref}}}
$$

| Cellular FOXP3 Expression | Applied Response |
|---|---|
| Below the reference value | Network response is proportionally reduced |
| Equal to the reference value | Original estimated network power is applied |
| Above the reference value | Network response is proportionally amplified |
| Zero or below | Target response scaling factor is set to zero |

The scaling factor is not capped when it exceeds the reference value. Therefore, cells with higher FOXP3 expression receive an amplified response, representing an extrapolation beyond the reference state used for network estimation.

### 3. Applying Signed Network Power to Cellular Expression

For each selected FOXP3 target gene $g$, the following expression change was applied once:

$$
x_{c,g}^{KO} = x_{c,g} + a_cP_{\mathrm{FOXP3},g}
$$

At the same time, FOXP3 itself was explicitly set to zero:

$$
x_{c,\mathrm{FOXP3}}^{KO}=0
$$

Consequently, negative power decreases target expression, whereas positive power increases it.

Expression levels of genes other than FOXP3 and the selected targets were retained unchanged. This procedure represents a **single application of signed changes to selected targets**, rather than recursive propagation through the network.

The original decoder outputs and the modified expression values were not clipped at zero. The occurrence of negative expression values was evaluated separately as a diagnostic.

### 4. Calculating the Direction of Change Using the Same Encoder

Both baseline and KO expression profiles were re-encoded using the same existing encoder. The difference between their posterior mean coordinates was calculated as follows:

$$
\Delta z_c = \mu(x_c^{KO})-\mu(x_c)
$$

The resulting displacement was projected onto the original PCA axes:

$$
\Delta p_c = \Delta z_cW_{\mathrm{PCA}}^\top
$$

Re-encoding the baseline expression as well allows common coordinate differences introduced by the decoder–encoder reconstruction process to be reduced, preventing them from being unnecessarily mixed with the predicted KO response.

### 5. Separating Direction and Response Magnitude in Arrow Visualization

The arrow direction was defined using the unit vector of the PCA displacement:

$$
u_c = \frac{\Delta p_c}{|\Delta p_c|_2}
$$

The arrow length was scaled according to the L2 magnitude of the actual target expression changes, rather than the PCA displacement itself:

$$
M_c = \sqrt{\sum_{g\in T}\left(a_cP_{\mathrm{FOXP3},g}\right)^2}
$$

$$
\vec{v}_c = sM_cu_c
$$

Here, $s$ is a common visualization scaling factor applied to all cells. In the current implementation, the maximum arrow length is set to 3% of the maximum PCA axis range, while preserving the relative response magnitude between cells.

The amount of FOXP3 expression removed is excluded from the calculation of $M_c$.

### Figure Interpretation

| Figure Element | Meaning |
|---|---|
| Blue points | Original PCA positions of observed CD4 Naive T cells |
| Red points | Original PCA positions of observed Regulatory T cells |
| Arrow starting point | Original PCA coordinate of each observed cell |
| Arrow direction | Predicted state-change direction estimated by re-encoding the virtual KO expression |
| Arrow length | L2 magnitude of the applied target expression changes |
| Short or absent arrows | Small target responses or absence of a measurable PCA displacement direction |

Cells with higher baseline FOXP3 expression receive relatively larger responses, whereas cells with lower FOXP3 expression receive smaller responses.

Therefore, the visualization illustrates **how the magnitude and direction of predicted cellular state changes differ across individual cells when the same FOXP3 KO response network is applied to different baseline cellular states**.

----
[Click here to view the Gene-Regulatory-Network (GRN) Construction notebook](https://github.com/AiJayce/scDQ-path/blob/Virtual-Trajectory-Construction/scDQN.ipynb)
----
