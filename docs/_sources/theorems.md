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

## {prf:ref}`theorem-PGFComp` — Composition of PGFs and randomly-stopped sums

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

## {prf:ref}`lem-PhiTimeSum` — The PGF of $\Phi(x,T_1+T_2)$

:::{admonition} Statement
:class: dropdown

Consider a continuous-time Galton-Watson Process with $X(t)$ representing the (random) population size at time $t$.  Then

$$
\Phi(x,T_1+T_2) = \Phi(\Phi(x,T_2),T_1)
$$
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

## {prf:ref}`theorem-LargeNSmallOutbreak` — Small outbreak size distribution ($N \to \infty$)

:::{admonition} Statement
:class: dropdown

In the limit $N \to \infty$, $I(t)$ for small outbreaks of continuous-time SIS and SIR diseases is indistinguishable from $X(t)$ in continuous-time Galton-Watson processes with $r_0 = \gamma$ and $r_2 = \beta$.
:::

## {prf:ref}`lemma-CycleLemma` — Cycle Lemma

:::{admonition} Statement
:class: dropdown

Given a sequence $S$ of $j$ non-negative integers summing to $j-1$, there is a unique tree whose Łukasiewicz word is one of the cyclic permutations of $S$.
:::

## {prf:ref}`theorem-ctsTimeSIS_SIR_SizeDist` — Continuous-time SIS and SIR small outbreak size distribution

:::{admonition} Statement
:class: dropdown

Consider the SIS and SIR disease models with transmission rate $\beta$ and recovery rate $\gamma$ and $\mathcal{R}_0 = \beta/\gamma$.  In the $N \to \infty$ limit, the probability an outbreak ends with exactly $\ell$ infections is

$$
\mathbb{P}[\ell \text{ infections}]=\frac{1}{\ell}\frac{\mathcal{R}_0^{\ell-1}}{(\mathcal{R}_0+1)^{2\ell-1}} \binom{2\ell-2}{\ell-1}
$$
:::

## {prf:ref}`thm-jointPGFProducts` — Composition of PGFs and randomly-stopped sums

:::{admonition} Statement
:class: dropdown

Given a PGF $\mu(x,y)$ for the joint distribution of a pair of non-negative integers $(X,Y)$.  

The PGF of the joint distribution of their sums $\sum_{i=1}^\ell (X_i, Y_i)$ is $\xi(x,y)^\ell$.  

If $\ell$ itself is a random variable with PGF $\psi(x)$, then the PGF for the joint distribution of the sums is $\psi(\xi(x,y))$.
:::

## {prf:ref}`thm-jointPGFComposition` — Composition of PGFs and sums of randomly-stopped sums

:::{admonition} Statement
:class: dropdown

Given two joint distributions of non-negative integers $p_{j,k}$ and $q_{j,k}$ with PGFs $\xi_1(x,y)$, $\xi_2(x,y)$.  

We take $\ell$ pairs $(X_i,Y_i)$, $i=1,\ldots, \ell$ from the first distribution and another $m$ pairs $(X_{i}, Y_{i})$, $i=\ell+1,\ldots, \ell+m$ from the second distribution.  The PGF of the sum $\sum_{i=1}^{\ell+m} (X_i,Y_i)$ of all of these pairs is $\xi_1(x,y)^\ell \xi_2(x,y)^m$.  

If in turn $\ell$ and $m$ are random variables whose joint distribution has PGF $\psi(x,y)$, then the PGF for the randomly-stopped sum is $\psi(\xi_1(x,y),\xi_2(x,y))$.
:::

## {prf:ref}`thm-BackwardTwoTypeGenBased` — PGF of size distribution of two-type Galton-Watson process at generation $g$ (Backward version)

:::{admonition} Statement
:class: dropdown

Given a two-type Galton-Watson process, with offspring distributions $\xi_1(x,y)$ and $\xi_2(x,y)$, the PGFs for the distribution at generation $g$ can be found by recursively solving

$$
\vec{\Phi}_{g+1}(x,y) = \left(\xi_1(\vec{\Phi}_{g}(x,y)),\quad \xi_2(\vec{\Phi}_{g}(x,y))\right)
$$
with

$$
\vec{\Phi}_0(x,y) = (x,y) 
$$
:::

## {prf:ref}`example-IRJointDistForward` — PGF of size distribution of two-type Galton-Watson process at generation $g$ (Forward version)

:::{admonition} Statement
:class: dropdown

Given a two-type Galton-Watson process, with offspring distributions $\xi_1(x,y)$ and $\xi_2(x,y)$, the PGFs for the distribution at generation $g$ can be found by recursively solving

$$
\vec{\Phi}_{g+1}(x,y) = \vec{\Phi}_g(\xi_1(x,y), \xi_2(x,y))
$$
with

$$
\vec{\Phi}_0(x,y) = (x,y)
$$
:::

## {prf:ref}`thm-TwoTypeExtinct` — Probability of complete extinction of two-type Galton-Watson Process

:::{admonition} Statement
:class: dropdown

The probabilities of extinction of both types by generation $g$, depending on initial condition is
$\vec{\alpha}(g) = (\alpha(g|(1,0)), \alpha(g|(0,1)))$.  This is found by 
iteratively solving

$$
\vec{\alpha}(g)= \left(\xi_1(\vec{\alpha}(g-1)), \quad\xi_2(\vec{\alpha}(g-1))\right)
$$
with $\vec{\alpha}(0) = (0,0)$.

The long-time extinction probability from each state solve $\vec{\alpha} = (\xi_1(\vec{\alpha}), \vec{\xi}_2(\vec{\alpha}))$.
:::

## {prf:ref}`thm-TwoTypeHalfExtinct` — Probability of extinction of one type in a two-type Galton-Watson Process

:::{admonition} Statement
:class: dropdown

If $\xi_2(x,y)$ can be written $\xi_2(y)$, then the probabilities of extinction of type $1$ by generation $g$ assuming an initial individual of type $1$ is $\alpha(g;1)$ where

$$
\alpha(g;1) = \xi_1(\alpha(g-1;1),1)
$$
with $\alpha(0;1) = 0$.

The symmetric result holds for $\alpha(g;2)$, the probability that the second type goes extinct assuming that $\xi_1(x,y) = \xi_1(x)$ and the initial individual is of type $2$.
:::
