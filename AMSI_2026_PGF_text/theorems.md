# List of Theorems

*(Auto-generated from source notebooks in `_toc.yml` order. To update: run `python generate_theorems_md_from_ipynb.py` and rebuild.)*

## {prf:ref}`thm-PGFLinComb` — Linear combination of PGFs

:::{admonition} Statement
:class: dropdown

Assume that we have a (possibly infinite) set of PGFs $\mu_1(x), \mu_2(x), \ldots$.  We choose a number $n$ by first choosing the distribution corresponding to $\mu_i(x)$ with probability $\pi_i$, and then we choose $n$ from the distribution.  The resulting distribution for $n$ has PGF

$$
\psi(x) = \sum_{i=1}^\infty \pi_i\mu_i(x)
$$
:::

## {prf:ref}`theorem-GFConvolution` — Convolution in series product

:::{admonition} Statement
:class: dropdown

Given two Generating Functions $f(x) = f_0 + f_1 x + f_2 x^2 + \cdots$ and $g(x) = g_0 + g_1 x + g_2 x^2 + \cdots$, then the coefficient of $x^n$ in the product $f(g)g(x)$ is given by 

$$
[x^n](f(x)g(x)) = \sum_{k=0}^n f_k g_{n-k}
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

## {prf:ref}`theorem-PGFRandSum` — PGF of random sum

:::{admonition} Statement
:class: dropdown

Given the PGFs

\begin{align*}
\mu_1(x) &= \sum_n p_n x^n\\
\mu_2(x) &= \sum_n q_n x^n
\end{align*}
If we choose $K$ from the distribution with PGF $\mu_1(x)$ and $M$ from the distribution with PGF $\mu_2(x)$, then the probability $\mathbb{P}[K+M=n]$ is 

$$
\mathbb{P}[K+M=n] = \sum_{k=0}^n p_k q_{n-k}
$$
and the PGF of the sum is

$$
\psi(x) = \mu_1(x)\mu_2(x)
$$
:::

## {prf:ref}`corrPGFprod` — Product of PGFs

:::{admonition} Statement
:class: dropdown

Given $k$ PGFs $\mu_1(x), \mu_2(x), \ldots, \mu_k(x)$, the product 

$$
\psi(x) = \prod_{i=1}^k \mu_i(x)
$$ 
is also a PGF and it corresponds to the probability distribution of the sum $n_1 + n_2 + \cdots + n_k$ where each $n_i$ comes from the distribution whose PGF is $\mu_i(x)$.
:::

## {prf:ref}`cor-PGFPower` — PGF of sum of repeated draws

:::{admonition} Statement
:class: dropdown

Given a PGF $\mu(x)$, if we perform $k$ draws from its distribution, the PGF for the sum of all $k$ draws is $\mu(x)^k$.
:::

## {prf:ref}`theorem-PGFComp` — Composition of PGFs

:::{admonition} Statement
:class: dropdown

Given PGFs $\mu_1(x)$ and $\mu_2(x)$, the composition $\psi(x) = \mu_1(\mu_2(x))$ is the PGF for the following process of choosing a number $n$:
- use the distribution with PGF $\mu_1(x)$ to choose a random integer $k$
- then choose $k$ numbers independently from the distribution with PGF $\mu_2(x)$ and add them together to get $n$.
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

If $\mu(x)$ is the PGF of the offspring distribution of a Galton-Watson process then $\Phi_g(x)$, the PGF of the distribution of $X_g$, satisfies

\begin{align*}
\Phi_0(x) &= x\\
\Phi_{g+1}(x) &= \mu(\Phi_g(x)) \quad g>0
\end{align*}
that is, $\Phi_0(x) = x$, &nbsp; $\Phi_1(x)=\mu(x)$, &nbsp; $\Phi_2(x)=\mu(\mu(x))$, &nbsp; $\cdots$, &nbsp;  $\Phi_g(x) = \mu^{(g)}(x) = \mu(\mu(\mu(\cdots \mu(x)\cdots)))$.
:::

## {prf:ref}`thm-ExpectedSize` — Average Size of a Galton Watson Process after $g$ generations

:::{admonition} Statement
:class: dropdown

The expected size of a Galton Watson process at generation $g$ is

$$
\mathbb{E}[X_g] = [\mu'(1)]^g
$$
:::

## {prf:ref}`theorem-Survival` — Survival Probability

:::{admonition} Statement
:class: dropdown

If an event occurs with rate $r$, the probability it has not happened after waiting a time $t$ is $e^{-rt}$.
:::

## {prf:ref}`lem-PGFatDt` — PGF of $X(t)$ at time $\Delta t$

:::{admonition} Statement
:class: dropdown

If $X(0)=1$, the PGF of the population size $X(t)$ at time $\Delta t \ll 1$ is

\begin{align*}
\Phi(x,\Delta t) &= (1-r\Delta t) x + (r \Delta t) \hat{\mu}(x) + \mathcal{o}(\Delta t)\\
 &= x + (r \Delta t) (\hat{\mu}(x)-x)+ \mathcal{o}(\Delta t)
\end{align*}
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
