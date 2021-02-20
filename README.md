# Latent Dirichlet Allocation
This repository implements Latent Dirichlet Allocation in Python 3, using strictly Python/NumPy routines.

Current version supports posterior inference via:
- Gibbs Sampling

I largely relied on the following two resources for implementing this algorithm:
- [```A Theoretical and Practical ImplementationTutorial on Topic Modeling and Gibbs Sampling```](https://u.cs.biu.ac.il/~89-680/darling-lda.pdf)
- [```Latent Dirichlet Allocation Using Gibbs Sampling```](https://ethen8181.github.io/machine-learning/clustering_old/topic_model/LDA.html)

Potential future to-do's:
- add black-box inference
- add basic pre-processing pipeline (stemming, stopword removal, regex)
- speed up topic inference / topic sampling procedure