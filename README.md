# Confido: Adaptive Trust Region Methods in JAX
Trust region optimisation in JAX!

Trust region algorithms consist principally of a Hessian approximation scheme, a method for choosing the trust region radius, and an algorithm for solving the trust region subproblem. Confido considers these as seperate components, with the algorithms for choosing the trust region radius and the algorithm for solving the trust region subproblem chosen at runtime. This is to allow for quick prototyping and testing adaptive trust region algorithms, ie. nonstandard methods of choosing the trust region size.

# TODO:
1. Create requirements file
2. Add proper axis scaling as regularisation method
3. Handle "the hard case"
4. Two-dim subspace method for solving subproblem

# Next steps
- choice of initial trust region size? start with this https://epubs.siam.org/doi/pdf/10.1137/S1064827595286955 and see how
to inculde some more sophisticated methods of automatically initialising the trust region.
- Other methods of trust region size update: 
    take inspo from 1st order gradient descent methods, feedback control theory, and
    existing "adaptive trust region radius" literature (which I am very skeptical of...)
    May be worth trying: neural network trust region updating. ie. network takes in data and outputs
    the size of the trust region. This might generalise to more problems than just performing optimisation
    via neural network to directly overstep the optimisation problem.
