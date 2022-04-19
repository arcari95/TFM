# EDAs
Implementation of Estimation of Distribution Algorithms (EDAs).

EDA is a framework to optimize black-box discrete optimization problems.  
The algorithm of EDA is as follows.
1. Initialize a population *P* whose size is &lambda;.
2. Construct a population *S* which includes promising solutions in *P*.
3. Build an explicit probabilistic model *M* based on *S*.
4. Generate new &lambda;<sub>candidate</sub> solutions from *M* to construct a population *O*.
5. The solutions in *P* is replaced with those of *O*.
6. If termination conditions are met, then the algorithm is terminated, else go to (2).
