# 4 Implementation and experiments

We implemented several agglomerative and divisive hierarchical clustering methods and compared their Dasgupta cost and Moseley–Wang revenue on Iris, Wine, and synthetic similarity graphs. Our experiments address three questions: how the practical methods compare on real data; whether a stronger Max-Uncut relaxation improves the resulting hierarchy on small instances; and which graph structures favour recursive partitioning over greedy agglomeration. We distinguish the algorithms analysed in the preceding sections from the practical approximations used in our implementation. The measurements provide evidence on the tested instances rather than proofs of worst-case approximation factors.

## 4.1 Implementation details

All algorithms were implemented in Python and executed using the project's `.venv` environment. We reused NumPy and SciPy for array operations, distances, eigendecomposition, and linkage; scikit-learn for feature standardization; and CVXPY with SCS for semidefinite optimization. The saved SDP experiments used CVXPY 1.9.3 and SCS 3.3.1. Trees use SciPy's linkage representation, which preserves the original observation indices. Our objective evaluator traverses the internal splits and counts the leaves beneath each lowest common ancestor; geometric linkage heights are not used as cluster sizes.

| Algorithm | Implementation |
|---|---|
| Single, complete, and average linkage | SciPy `cluster.hierarchy.linkage`. We supply condensed dissimilarities $d_{ij}=C-w_{ij}$, where $C=\max_{i,j}w_{ij}$. This implements maximum-, minimum-, and mean-similarity merging, respectively. |
| Random binary tree | Custom recursion assigns vertices independently to two sides with probability $1/2$, repeats empty splits, and recurses until singleton leaves. Splits are not forced to be balanced. |
| Recursive spectral cut | Form the induced unnormalized Laplacian $D-W$, compute a Fiedler vector with SciPy, sort its coordinates, and sweep all nonempty prefixes. Choose the split minimizing $w(A,B)/(\lvert A\rvert\lvert B\rvert)$ and recurse. Disconnected components are separated without cutting positive edges. |
| Recursive basic-SDP Max-Uncut | Solve a basic SDP at each nontrivial node, perform Gaussian rounding and greedy balancing, retain the best of 32 candidates, and recurse on both parts. |
| Root Max-Uncut + average linkage | Apply the basic-SDP bisection once at the root, then run average linkage independently within the two halves. |
| Root Max-Uncut + average, best-of | Compare the preceding candidate with global average linkage and return the tree with higher revenue. Ties select global average linkage. |
| Root Max-Uncut + random splits | Apply the basic-SDP bisection once at the root, then generate independent random binary subtrees within the two halves. |

The spectral routine is a **heuristic for the uniform sparsest-cut objective**, not the certified sparsest-cut subroutine from the cost-approximation analysis. In particular, we did not implement the Arora–Rao–Vazirani algorithm. We therefore use “recursive spectral” in the tables rather than assigning this implementation the theoretical recursive sparsest-cut approximation factor.

For an even-sized subproblem, the basic Max-Uncut SDP is

$$
\max_X\;\frac14\sum_{i,j}w_{ij}(1+X_{ij})
\quad\text{subject to}\quad
X\succeq0,\quad X_{ii}=1,\quad X\mathbf1=\mathbf0.
$$

The matrix represents pairwise sign correlations. We use zero first moments, giving zero Gaussian thresholds, and round the vectors to signs. If the partition is unbalanced, we repeatedly move the vertex with the smallest net loss of uncut weight from the larger side. Odd-sized subproblems are padded with a zero-weight dummy vertex, which is removed after balancing. Two-vertex and uniform-weight subproblems use an analytic split. SCS uses tolerance $10^{-5}$ and at most 20,000 iterations. This tolerance controls numerical optimization accuracy; it is distinct from the approximation parameter $\varepsilon$ in the theoretical hierarchy algorithm. Small numerical PSD and diagonal errors are repaired before sampling, and feasibility residuals are saved. The numerical SDP objective is not treated as a certified bound.

### Practical limitations of the theoretical Max-Uncut algorithm

The stronger Max-Uncut guarantees require more than the basic correlation SDP. Wu, Du, and Xu's SDPH-GB algorithm uses a high-level Lasserre relaxation, conditioning, a vector modification, Gaussian threshold rounding, and size adjustment. Its explicit linear-bias routine uses $\theta=0.87$ and reports the nominal factor 0.8732; the separate 0.8776 claim involves the pairing-based rounding discussed earlier. We did not implement that pairing variant.

The hierarchy depth is the main obstacle to a direct large-instance reproduction. The paper sets

$$
t=\left(20\left\lceil1/\varepsilon\right\rceil\right)^{12}
$$

and solves a level-$(t+1)$ hierarchy. Even $\varepsilon=0.1$ gives $t=4.096\times10^{27}$. The polynomial-time statement treats $\varepsilon$ as fixed, with an impractically large dependence on it. A full finite Boolean hierarchy can instead be represented exactly using exponentially many assignments, which is still unsuitable for the complete Iris and Wine inputs. Consequently, our large-instance Max-Uncut methods use the basic SDP and **do not inherit the stronger published revenue or bisection guarantees**. The best-of average-linkage construction retains the average-linkage guarantee because it never returns a lower-revenue tree than that baseline.

For the 6- and 8-vertex experiments, we implemented a separate full-hierarchy calculation. At full level, Boolean moment positivity is equivalent to nonnegative assignment probabilities, and lifted balance constraints restrict support to balanced assignments. We solve the resulting probability LP with SciPy HiGHS and check its value by exhaustive enumeration. To select an optimal hierarchy solution, we randomly choose an optimal complementary pair of assignments and put equal mass on its two orientations. Conditioning on one randomly selected vertex and its marginal sign identifies the orientation; subsequent conditioning steps leave the point mass unchanged. This is an exact but highly specialized finite-instance implementation, using exponential optimization rather than a scalable truncated hierarchy.

We then apply the configured linear-bias rounding and balancing, integrating over every conditioning endpoint and every rounded sign outcome. After integral conditioning, each sign flips independently with probability $(1-0.87)/2=0.065$ before balancing. Thus this pipeline can lose objective value even though its pre-rounding assignment is optimal. We also solved actual level-1 and level-2 moment SDPs for comparison; their solutions were not substituted for the required high-level hierarchy.

These small-instance calculations are diagnostic rather than a complete certification of the paper's approximation analysis. In particular, the normalized-correlation expression printed in Wu et al.'s Lemma 2(d) includes diagonal terms. After zero residuals are replaced with unit directions, those diagonal terms remain one, so the printed small-correlation condition is not verified on these tiny inputs. The detailed qualification is recorded in [the hierarchy audit and results](SMALL_HIERARCHY_RESULTS.md).

## 4.2 Experimental setup

### Real datasets and preprocessing

We used the [Iris](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) and [Wine](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data) datasets from the UCI Machine Learning Repository. Wine here is the 178-observation classification dataset, not Wine Quality. Iris has four numerical features and a species label; Wine has thirteen numerical features and a class label. Labels were retained separately and excluded from feature transformations, similarities, tree construction, and objective evaluation.

For each dataset, we standardized every numerical feature using its mean and population standard deviation:

$$
z_{ik}=\frac{x_{ik}-\mu_k}{s_k}.
$$

This prevents features with larger numerical scales from dominating Euclidean distance. We then calculated pairwise distances between the standardized observations and chose the Gaussian bandwidth as the median strictly positive distance over unordered pairs:

$$
\sigma=\operatorname{median}\{\|z_i-z_j\|_2:i<j,\ \|z_i-z_j\|_2>0\}.
$$

The similarity graph uses

$$
w_{ij}=\exp\left(-\frac{\|z_i-z_j\|_2^2}{2\sigma^2}\right)\quad(i\ne j),
\qquad w_{ii}=0.
$$

All numerical entries were finite. Duplicate observations were retained, and no PCA, imputation, or train/test split was applied: the full dataset is the unsupervised clustering instance. We used a single bandwidth rule consistently rather than tuning bandwidth to favour a particular algorithm.

| Dataset | Observations | Numerical features | Bandwidth $\sigma$ | Total weight $W$ | Revenue upper bound $U$ |
|---|---:|---:|---:|---:|---:|
| Iris | 150 | 4 | 2.49764591 | 6,731.399492 | 996,247.12 |
| Wine | 178 | 13 | 5.00351340 | 9,709.014522 | 1,708,786.56 |

### Synthetic graph construction

We focus on **two synthetic families**: balanced nested graphs and the cross-pair stress family. Both directly specify pairwise similarities; they are not feature vectors and are not passed through standardization or the Gaussian similarity kernel. They provide contrasting structural tests, rather than a claim that either agglomerative or divisive methods always dominate.

| Family | Size and instances | Structure before noise | Stylized real-world interpretation |
|---|---|---|---|
| Balanced nested | One graph with 128 vertices | Three binary levels with eight groups of 16. A pair sharing $k$ levels has weight $0.1+0.8k/3$. | An organization with similarly sized divisions, departments, and teams, with stronger interaction inside smaller units. |
| Cross-pair stress | Ten graphs with 32 vertices each | Two groups of 16; within-group weight 0.6, a cross-group perfect matching of weight 1.0, and other cross-group weights 0.02. | Two communities with strong individual partnerships across their boundary, potentially obscuring the broader community structure. |

The balanced nested graph uses generation seed 4234 and independent Gaussian weight noise with standard deviation 0.02. The cross-pair graphs use seeds 5000–5009 and noise standard deviation 0.01. In both cases, noise is added once per unordered pair; weights are clipped to $[0,1]$, mirrored, and given zero diagonal. Vertex order is shuffled. Planted labels and noise-free matrices are available for inspection but are not algorithm inputs or certificates of optimality.

The cross-pair family was deliberately introduced as an exploratory stress test after earlier experiments. Its strongest individual edges conflict with the collective similarity within each group. The balanced nested graph instead provides a clean hierarchical structure with compatible balanced splits. We expect linkage to be competitive on this structure, but do not describe it as a demonstrated large agglomerative advantage: the measured recursive methods are nearly tied with average linkage.

![The two synthetic similarity structures](results/report_figures/synthetic_structures.png)

*Figure 4.1. The balanced nested graph (left) and one cross-pair stress graph (right), reordered into construction order for visualization. Algorithms receive shuffled matrices. The two families have different graph sizes; each method is compared on exactly the same input within a family.*

### Objectives, normalization, and repetitions

For total edge weight $W=\sum_{i<j}w_{ij}$, we evaluate

$$
C(T)=\sum_{i<j}w_{ij}|T_{ij}|,
\qquad
R(T)=\sum_{i<j}w_{ij}(n-|T_{ij}|).
$$

Lower cost and higher revenue are preferable. Since $C(T)+R(T)=nW$, the two objectives give the same ranking on a fixed graph, although their approximation ratios differ. We therefore use revenue for the principal plots and include costs in the tables.

For the full real and synthetic datasets, exact optimum is unavailable. We normalize revenue by

$$
U=(n-2)W\geq\mathrm{OPT}_R.
$$

Thus $R/U$ is a **lower bound** on $R/\mathrm{OPT}_R$, not an exact approximation ratio. A measured value at least $1/3$ certifies that ratio for the particular returned tree; observing values on finitely many inputs cannot establish a universal guarantee. The fair-coin random-tree distribution has exact expected revenue $U/3$.

| Experiment group | Graphs | Repetitions |
|---|---|---|
| Full Iris, Wine, and balanced nested graph | Three fixed graphs | Linkage and spectral once per graph; random trees use 100 seeds (4234–4333); recursive basic SDP and root constructions use 20 seeds (4234–4253). |
| Small exact-OPT comparison | Six graphs with 6 or 8 vertices | Deterministic methods once; basic-SDP methods use 20 seeds (0–19). Global-random and full-hierarchy expectations are computed analytically or by complete finite enumeration. |
| Exploratory cross-pair family | Ten independent noisy 32-vertex graphs | Deterministic methods once per graph; random and recursive basic SDP use 20 seeds (0–19) per graph. |

The reported small suite contains a reproducible 6-point and 8-point subset of Iris, Wine, and the balanced nested graph, giving six cases. The 32-vertex cross-pair graphs are not part of the exact-OPT experiment. Earlier, broader synthetic and diagnostic results remain archived but are excluded from the tables, figures, and small-instance averages in this section. Subsets retain the existing similarity weights: feature scaling and bandwidth are not refitted. We calculate optimal bisection values by enumeration and optimal tree cost and revenue by a subset dynamic program with $O(3^n)$ time and $O(2^n)$ memory. Small-instance ratios therefore use true optimal revenue. “Exact” expectation means complete enumeration in floating-point arithmetic, rather than Monte Carlo estimation or an interval-arithmetic certificate.

For the basic-SDP root + random construction on small inputs, we average over the 20 sampled root partitions and integrate the random-subtree revenue conditional on each root. On the full datasets, root + random results are the means of the actual sampled trees. Standard deviations in the real-data plot reflect algorithm randomness on fixed data. In the exploratory-family plot, standard deviations are instead computed across the ten graph-level means. Neither should be interpreted as a universal performance interval. Timings were not collected under a controlled hardware and concurrency protocol, so we do not use them to rank methods.

## 4.3 Experimental results

### 4.3.1 Iris and Wine

Table 4.1 reports the completed real-data results. The Max-Uncut entries use the current basic-SDP implementation; historical NetworkX results are excluded. Values for repeated algorithms are arithmetic means.

| Dataset | Algorithm | Runs | Cost ↓ | Revenue ↑ | Revenue / U ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Iris | Single linkage | 1 | 554,198.70 | 455,511.22 | 0.457227 |
| Iris | Complete linkage | 1 | 568,831.01 | 440,878.92 | 0.442540 |
| Iris | Average linkage | 1 | 549,101.28 | 460,608.64 | 0.462344 |
| Iris | Random tree | 100 | 677,457.05 | 332,252.87 | 0.333504 |
| Iris | Recursive spectral | 1 | 554,317.59 | 455,392.33 | 0.457108 |
| Iris | Recursive basic SDP | 20 | 571,271.57 | 438,438.35 | 0.440090 |
| Iris | Root SDP + average | 20 | 567,467.27 | 442,242.65 | 0.443909 |
| Iris | Root SDP + average (best-of) | 20 | 549,101.28 | 460,608.64 | 0.462344 |
| Iris | Root SDP + random | 20 | 585,980.20 | 423,729.72 | 0.425326 |
| Wine | Single linkage | 1 | 1,071,393.68 | 656,810.91 | 0.384373 |
| Wine | Complete linkage | 1 | 1,083,382.12 | 644,822.46 | 0.377357 |
| Wine | Average linkage | 1 | 1,051,952.34 | 676,252.25 | 0.395750 |
| Wine | Random tree | 100 | 1,158,372.44 | 569,832.15 | 0.333472 |
| Wine | Recursive spectral | 1 | 1,057,592.78 | 670,611.81 | 0.392449 |
| Wine | Recursive basic SDP | 20 | 1,061,424.81 | 666,779.78 | 0.390207 |
| Wine | Root SDP + average | 20 | 1,056,580.62 | 671,623.97 | 0.393041 |
| Wine | Root SDP + average (best-of) | 20 | 1,051,952.34 | 676,252.25 | 0.395750 |
| Wine | Root SDP + random | 20 | 1,078,698.38 | 649,506.20 | 0.380098 |

*Table 4.1. Cost, revenue, and normalized revenue on the complete real datasets. “Root” means that Max-Uncut is used only for the first partition. Best-of compares the root + average candidate with global average linkage.*

![Real-data revenue comparison](results/report_figures/real_data_revenue.png)

*Figure 4.2. Revenue divided by $U$ on Iris and Wine. Error bars show one sample standard deviation across algorithm seeds; deterministic methods have no error bar. Several error bars are smaller than the plotting resolution. The dashed line is the exact expected random-tree baseline $1/3$, not an estimate of OPT.*

Average linkage obtains the highest revenue and lowest cost on both real datasets. Its normalized revenues are 0.462344 on Iris and 0.395750 on Wine, both above $1/3$. These values establish the desired ratio on these two instances even without knowing OPT. Single linkage is narrowly ahead of spectral cutting on Iris, while spectral cutting is the strongest non-average baseline on Wine. Recursive basic SDP improves substantially over random trees but does not overtake average linkage.

The root + average candidate is weaker than global average linkage on both datasets, so the best-of version returns the latter. Using the same root followed by random splits gives lower revenue than using average-linkage subtrees. The random-tree means are close to $U/3$, consistent with the theoretical expectation. These observations do not contradict stronger worst-case results for partition-based algorithms: approximation guarantees bound their performance on arbitrary inputs, rather than require them to beat average linkage on every input. Moreover, our full-dataset partition solvers do not implement all the components needed for those guarantees.

### 4.3.2 Small datasets and the hierarchy-based Max-Uncut calculation

Table 4.2 compares all methods on the same six small graphs: the 6- and 8-point subsets of Iris, Wine, and balanced nested. Every case receives equal weight. The minimum is the lowest case-level expected or mean ratio, not the worst individual random draw. These aggregates have been recalculated for the selected cases rather than copied from the earlier, broader benchmark.

| Algorithm | Mean revenue / OPT | Minimum case ratio |
| --- | ---: | ---: |
| Single linkage | 0.999960 | 0.999759 |
| Complete linkage | 0.992080 | 0.952722 |
| Average linkage | 0.999960 | 0.999759 |
| Recursive spectral | 0.999834 | 0.999243 |
| Random tree | 0.698870 | 0.519459 |
| Recursive basic SDP | 0.910620 | 0.825972 |
| Basic SDP root + average | 0.917438 | 0.857788 |
| Full hierarchy root + average | 0.909422 | 0.844579 |
| Basic SDP root + random | 0.884490 | 0.787913 |
| Full hierarchy root + random | 0.876154 | 0.774650 |
| Basic SDP + average (best-of) | 0.999960 | 0.999759 |
| Full hierarchy + average (best-of) | 0.999960 | 0.999759 |

*Table 4.2. Revenue divided by exact optimal revenue on the six selected small graphs. Basic-SDP methods use the best of 32 Gaussian candidates per bisection; full-hierarchy results integrate one Gaussian candidate after conditioning. These are different rounding budgets.*

![Exact-OPT comparison on the six selected small graphs](results/report_figures/small_exact_comparison.png)

*Figure 4.3. Mean and range of case-level revenue / OPT across the six selected graphs. Points show the unweighted mean and line segments show the minimum-to-maximum range across cases, not confidence intervals. Orange points identify the full-hierarchy variants.*

Average linkage and single linkage are already near optimal on these small inputs, with mean ratio 0.999960. Both best-of constructions match global average linkage on every case. Recursive spectral cutting is also strong, with mean ratio 0.999834. This selected suite is therefore easy for several methods; it does not demonstrate a unique advantage of average linkage or the Gaussian similarity kernel.

The basic-SDP root routine reaches the exact bisection optimum, within numerical precision, in all 120 sampled runs across these six cases. The full-hierarchy routine also finds an optimal bisection before rounding, but its average expected ratio after rounding and balancing is 0.990358; the lowest case is 0.983166. Its downstream mean revenue ratios, 0.909422 with average-linkage subtrees and 0.876154 with random subtrees, are consequently below the corresponding basic-SDP values of 0.917438 and 0.884490. This difference reflects the **rounding stage and unequal candidate counts**, rather than evidence that the weaker relaxation optimizes better. A controlled comparison of one versus 32 attempts for both methods has not yet been performed.

On these six cases, the level-2 moment SDP agrees with exact bisection OPT within $10^{-6}$. This empirical tightness does not establish that level 2 is sufficient in general or supplies the conditioning requirements of Wu's analysis. The full-hierarchy calculation is still used for the reported conditioning experiment.

Small graph size also matters when interpreting the revenue approximation. For a balanced root with parts of size $n/2$, the exact expected revenue of subsequent fair-coin random subtrees is

$$
\mathbb E[R\mid A,B]=\frac{2(n-1)}{3}\bigl(W(A)+W(B)\bigr).
$$

Combining this with the bisection comparison $\mathrm{OPT}_R\leq n\,\mathrm{OPT}_{\mathrm{uncut}}$ gives a finite-size transfer factor $2(n-1)p/(3n)$ for a bisection expected ratio $p$. We retain this endpoint correction rather than applying the limiting $2p/3$ expression directly. The underlying derivation and separate diagnostic examples remain available in [the small-instance audit](SMALL_HIERARCHY_RESULTS.md).

### 4.3.3 Balanced nested family

The balanced nested graph tests how the methods handle an explicit hierarchy whose groups split evenly. Table 4.3 includes the six main methods and the three basic-SDP root constructions.

| Algorithm | Mean cost ↓ | Mean revenue ↑ | Revenue / U ↑ |
| --- | ---: | ---: | ---: |
| Single linkage | 130,954.28 | 211,171.45 | 0.627031 |
| Complete linkage | 130,948.10 | 211,177.63 | 0.627049 |
| Average linkage | 130,941.21 | 211,184.52 | 0.627070 |
| Random tree | 229,934.76 | 112,190.98 | 0.333128 |
| Recursive spectral | 130,944.76 | 211,180.98 | 0.627059 |
| Recursive basic SDP | 130,942.31 | 211,183.42 | 0.627066 |
| Root SDP + average | 130,941.21 | 211,184.52 | 0.627070 |
| Root SDP + average (best-of) | 130,941.21 | 211,184.52 | 0.627070 |
| Root SDP + random | 150,433.60 | 191,692.13 | 0.569191 |

*Table 4.3. Results on the 128-vertex balanced nested graph. Lower cost and higher revenue are preferable. Repeated-method values are means; revenue is normalized by the upper bound $U$, not exact OPT.*

Average linkage, recursive basic SDP, and spectral cutting are nearly tied, with normalized revenues of 0.627070, 0.627066, and 0.627059, respectively. Single and complete linkage are also close. The result is consistent with both agglomerative and divisive methods recovering a coherent hierarchy, and does not show a practically meaningful advantage for either category. Root + average linkage matches global average linkage here, whereas random subtrees reduce revenue.

![Comparison on the two selected synthetic families](results/report_figures/synthetic_performance.png)

*Figure 4.4. Revenue / $U$ for the six main methods on the two selected families. Balanced nested uses one fixed graph; cross-pair stress uses means across ten graph instances. Comparisons within each row use identical inputs for all methods. The two rows have different graph sizes and potentially different upper-bound tightness, so their absolute ratios should not be interpreted as a ranking of dataset difficulty.*

### 4.3.4 A family that favours recursive partitioning

To investigate a favourable regime explicitly, we tested the additional balanced-block family with stronger cross-group matching edges described in Section 4.2. The ten generated instances were evaluated with the same six main algorithms. Table 4.4 first averages the randomized methods within each graph, then averages across the ten graph instances; its SD measures variation across graph-level means.

| Algorithm | Mean cost ↓ | Mean revenue ↑ | Mean revenue / U | SD across graph means |
| --- | ---: | ---: | ---: | ---: |
| Single linkage | 3,400.75 | 1,872.38 | 0.378752 | 0.000313 |
| Complete linkage | 3,399.78 | 1,873.35 | 0.378947 | 0.000429 |
| Average linkage | 3,395.51 | 1,877.62 | 0.379812 | 0.000272 |
| Random tree | 3,625.71 | 1,647.42 | 0.333245 | 0.002490 |
| Recursive spectral | 2,291.47 | 2,981.66 | 0.603140 | 0.000555 |
| Recursive basic SDP | 2,291.41 | 2,981.72 | 0.603153 | 0.000562 |

*Table 4.4. Exploratory cross-pair family, $n=32$, ten independent noise/shuffle seeds. Exact OPT is not computed; normalized revenues use $U$. This family was deliberately constructed after inspecting the original benchmark.*

![Exploratory family and results](results/report_figures/cross_pair_stress_test.png)

*Figure 4.5. Left: one noisy input in construction order. The off-diagonal matching contains the strongest individual edges. Right: mean normalized revenue and one SD across ten graph-level means; the small SDs may be difficult to distinguish at this scale.*

Recursive basic SDP and spectral cutting achieve mean normalized revenues of 0.603153 and 0.603140, respectively, compared with average linkage's 0.379812. Their mean revenues are approximately **58.8% higher**, and their costs are approximately **32.5% lower**, than average linkage's. Both outperform average linkage on every generated graph. The very small difference between the two recursive methods is not evidence of a meaningful superiority of one over the other.

The saved trees reveal the mechanism. On every graph, average linkage's first sixteen merges follow the strongest edges across the two groups, creating mixed pairs that cannot later be separated by an agglomerative method. By contrast, spectral cutting recovers the planted groups at the root on all ten graphs, and recursive basic SDP does so in all 200 runs. A global cut can weigh the total similarity within each dense group against the relatively few stronger edges crossing between them. This example therefore distinguishes locally attractive merges from globally favourable partitions.

The evidence supports **balanced dense groups with sparse, stronger cross-group pairs** as a family favourable to our two recursive heuristics. For the spectral routine, this is evidence about the implemented sparsity sweep, not a test of the certified recursive sparsest-cut algorithm. The experiment also remains narrow: it covers one parameter setting, one graph size, and ten perturbations of a deliberately chosen structure. It demonstrates the existence of a favourable regime rather than typical superiority on arbitrary similarity data.

A subsequent validation also tested the **root-only basic-SDP + average, best-of** construction on these ten graphs, using twenty seeds per graph. It improves over global average linkage in all 200 runs, increasing mean revenue from 1,877.62 to 2,981.81 (a mean relative gain of 58.81%). Independent pairwise scoring verifies that its returned revenue equals the maximum of the candidate and global average linkage. The selector therefore behaves as intended: it guarantees non-decreasing revenue, not strict improvement on every instance. Detailed checks are in [the best-of validation](BEST_OF_VALIDATION.md).

### 4.3.5 Interpretation and limitations

The experiments show that theoretical approximation factors and empirical rankings answer different questions. Average linkage is strong on Iris, Wine, and the balanced nested graph, while the cross-pair stress family exposes a substantial advantage for global recursive cuts. A stronger SDP relaxation alone is insufficient to predict the final revenue because rounding, balancing, and the choice of subtree algorithm also matter.

The median-bandwidth Gaussian kernel is one fixed modelling choice for Iris and Wine; we have not performed a bandwidth sensitivity study. Both selected synthetic families bypass that kernel, so their results cannot be attributed to its bandwidth. Additional noise levels, graph sizes, independent balanced nested instances, and matched rounding budgets would be needed for broader conclusions. We also did not implement the report's spreading-metric LP lower bound or certified sparsest-cut solver. On small inputs we used exact tree OPT instead; on large inputs we report only the simple revenue upper bound.

The current test suite passes 93 tests. All 440 trees from the new exploratory family were independently rescored from their cophenetic leaf counts and checked against their saved sources. The six selected small cases draw from previously validated tree archives; their report aggregates were recomputed from the individual case summaries. Source hashes, seeds, solver diagnostics, and individual scores are preserved alongside the summaries. These checks support reproducibility and implementation consistency, not a proof of the theoretical approximation guarantees.

### Reproducibility and source files

- Full-data results: [real-data comparison](results/revenue_comparison_sdp.csv), [archived synthetic scores, filtered to balanced nested here](results/synthetic_sdp/scores.csv), and [root-construction scores](results/root_bisection/scores.csv).
- Small-instance results: [full-hierarchy expectations](results/wu_hierarchy_small/expectations.csv), [moment-SDP diagnostics](results/wu_hierarchy_small/hierarchy_relaxations.csv), and [selected-case comparison](results/report_figures/small_selected_summary.csv).
- Exploratory family: [graph-level results](results/exploratory_cross_pairs/per_graph.csv), [metadata](results/exploratory_cross_pairs/metadata.json), and [validation](results/exploratory_cross_pairs/validation.json).
- Implementation: [preprocessing](src/hierarchical_clustering/preprocessing.py), [basic SDP](src/hierarchical_clustering/max_uncut.py), [full hierarchy](src/hierarchical_clustering/wu_hierarchy.py), [spectral sweep](src/hierarchical_clustering/spectral.py), and [exploratory generator and runner](src/hierarchical_clustering/cross_pair_experiments.py).
- Algorithm sources: [Wu, Du, and Xu (2015), local full text](max-uncut.pdf), [Austrin, Benabbas, and Georgiou, local full text](austrine.pdf), and [Alon, Azar, and Vainstein (2020)](https://proceedings.mlr.press/v125/alon20b/alon20b.pdf). The latter's finite-size transfer calculation is qualified above rather than assumed directly.

The existing measurements are sufficient for the two-family report. To regenerate this section and its figures without rerunning experiments:

```bash
.venv/bin/python scripts/build_experiment_report.py
```

The report builder reads the saved experiment outputs and filters them to the stated scope. Historical datasets and broader experiment results remain unchanged. Each embedded PNG has a corresponding SVG in `results/report_figures/` for inclusion in the final paper.
