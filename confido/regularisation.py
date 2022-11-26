import jax.numpy as jnp

###
# The hessian is a symmetric matrix and thus n x n
# each of these regularisation methods take the size n of
# the Hessian and optional extra args and returns a matrix D such that
# the trust region subproblem is ||D(update)|| < trust region radius.
# The standard method used (in LM) is not to have D be an appropriate
# rescaling of the axes. We still refer to this as a "regularisation"
# and no regularisation is D = I, the identity.
###


def no_regularisation(hessian_dim: int):
    return jnp.eye(hessian_dim)


def second_diff(hessian_dim: int):
    """
    Return central second difference operator
    """
    A = (
        jnp.diag(-jnp.ones(hessian_dim - 1), -1)
        + jnp.diag(2 * jnp.ones(hessian_dim), 0)
        + jnp.diag(-jnp.ones(hessian_dim - 1), 1)
    )
    # TODO: Check boundary conditions compared to other implementation
    A = A.at[0, 0].set(1)
    A = A.at[-1, -1].set(1)
    return A


def classic_lm(hessian_dim: int):
    pass
