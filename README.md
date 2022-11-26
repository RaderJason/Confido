# Confido
Trust region optimisation in JAX

# TODO:
1. Create requirements file
5. Add proper axis scaling as regularisation method
7. Add Cauchy point solve of subproblem
8. Add Newton trust region method
9. Add dogleg solve of subproblem

# Next steps
- choice of initial trust region size? start with this https://epubs.siam.org/doi/pdf/10.1137/S1064827595286955 and see how
to inculde some more sophisticated methods of automatically initialising the trust region.
- Other methods of trust region size update: 
    take inspo from 1st order gradient descent methods, feedback control theory, and
    existing "adaptive trust region radius" literature (which I am very skeptical of...)
    May be worth trying: neural network trust region updating. ie. network takes in data and outputs
    the size of the trust region. This might generalise to more problems than just performing optimisation
    via neural network to directly overstep the optimisation problem.