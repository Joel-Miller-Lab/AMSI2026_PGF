# List of Theorems

*(Auto-generated from source notebooks in `_toc.yml` order. To update: run `python generate_theorems_md_from_ipynb.py` and rebuild.)*

## {prf:ref}`thm-PGFLinComb` — Linear combination of PGFs

:::{admonition} Statement
:class: dropdown

Assume that we have a (possibly infinite) set of PGFs $\mu_1(x), \mu_2(x), \ldots$.  We choose a number $n$ by first choosing the PGF $\mu_i(x)$ with probability $\pi_i$, and then we choose $n$ from the distribution with PGF $\mu_i(x)$.  The resulting distribution for $n$ has PGF

$$
\psi(x) = \sum_{i=1}^\infty \pi_i\mu_i(x)
$$
:::

## {prf:ref}`thm-partialsums` — Partial Sums

:::{admonition} Statement
:class: dropdown

Consider a sequence of numbers $c_0, c_1, \ldots$ whose generating function is $f(x)$.  The partial sums 

\begin{align*}
s_0 &= c_0\\
s_1 &= c_0 + c_1\\
\end{align*}
and in general

$$
s_n = \sum_{i=0}^n c_i
$$
have generating function $\frac{f(x)}{1-x}$
:::

## {prf:ref}`thmPGFprod` — Product of PGFs

:::{admonition} Statement
:class: dropdown

Given $k$ PGFs $\mu_1(x), \mu_2(x), \ldots, \mu_k(x)$, the product 

$$
\psi(x) = \prod_{i=1}^k \mu_i(x)
$$ 
is also a PGF
and it corresponds to the probability distribution of the sum $n_1 + n_2 + \cdots + n_k$ where each $n_i$ comes from the distribution whose PGF is $\mu_i(x)$.
:::

## {prf:ref}`cor-PGFPower` — Sum of repeated draws

:::{admonition} Statement
:class: dropdown

Given a PGF $\mu(x)$, if we perform $k$ draws from its distribution, the PGF for the sum of all $k$ draws is $\mu(x)^k$.
:::

## {prf:ref}`thm-GWTisGWP` — Galton-Watson trees are Galton-Watson processes.

:::{admonition} Statement
:class: dropdown

Given a Galton-Watson tree, the sequence of numbers $\{X_g\}$, where $X_g$ is the number of nodes in generation $g$, forms a Galton-Watson process.  Similarly, every Galton-Watson process can be represented by a Galton-Watson tree.
:::

## {prf:ref}`thm-GWExtinctThresh` — Galton-Watson extinction threshold

:::{admonition} Statement
:class: dropdown

Given a Galton-Watson process whose offspring distribution has PGF $\mu(x)$:
- If $\mu'(1)<1$, then extinction is guaranteed.  
- If $\mu'(1)>1$, then long-term persistence (that is, never going extinct) is possible.
:::

## {prf:ref}`theorem-GW-size-by-generation` — Galton-Watson sizes at later generations

:::{admonition} Statement
:class: dropdown

If $\mu(x)$ is the PGF of the offspring distribution of a Galton-Watson process, then the distribution of sizes at generation $g$ has PGF $\mu^{(g)}(x)$ where $\mu^{(0)}(x) = x$ and $\mu^{(g)}(x) = \mu^{(g-1)}(\mu(x)) = \mu(\mu^{(g-1)}(x))$.
:::

## {prf:ref}`thm-ExpectedSize` — Average Size of a Galton Watson Process after $g$ generations

:::{admonition} Statement
:class: dropdown

The expected size of a Galton Watson process at generation $g$ is

$$
\mathbb{E}[X_g] = [\mu'(1)]^g
$$
:::

## {prf:ref}`InfDisGaltWat` — Early stages of infectious-disease outbreaks are Galton-Watson Processes

:::{admonition} Statement
:class: dropdown

If:
- an infectious disease outbreak begins with a single infected individual (the *index case* ) counted as generation $0$, 
- the infections are separated by generation so that those infected by generation $g$ are considered to be in generation $g+1$,
- the number of transmissions caused by each infected individual is independent of all others and has PGF $\mu(x)$.
- the population is large enough that every transmission goes to a new never-before infected individual,

Then the number of infections at generation $g$ forms a Galton-Watson process with offspring distribution $\mu(x)$.  If we generate edges from infector to infectee, we get a Galton-Watson tree.
:::

## {prf:ref}`theorem-TotalSizeDist` — Distribution of Total Sizes of Galton-Watson Processes

:::{admonition} Statement
:class: dropdown

Given a Galton-Watson Process whose offspring distribution PGF is $\mu(x)$, the probability that the process terminates after a finite cumulative count of exactly $\sum_{g=0}^\infty X_g = j$ is given by 

$$ \mathbb{P}\left(j=\sum X_g\right) = \frac{1}{j} p_{j-1}^{(j)}$$ 

where $p_{j-1}^{(j)}= [x^{j-1}]\left(\mu(x)^j\right)$ denotes the coefficient of $x^{j-1}$ in  $\mu(x)^j$.
:::

## {prf:ref}`lemma-CycleLemma` — Cycle Lemma

:::{admonition} Statement
:class: dropdown

Given a sequence of $j$ non-negative integers summing to $j-1$, there is exactly one cyclic permutation of the sequence that is a Łukasiewicz word.
:::
