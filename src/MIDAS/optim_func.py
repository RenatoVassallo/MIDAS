import numpy as np

def ssr(a, x_list, y, yl, weight_methods):
    error = y - a[0]  # intercept
    offset = 1
    num_regressors = len(x_list)

    # Betas
    betas = a[offset:offset + num_regressors]
    offset += num_regressors

    # Each regressor's weight parameters
    for x, beta, method in zip(x_list, betas, weight_methods):
        num_params = method.num_params
        theta = a[offset:offset + num_params]
        xw, _ = method.x_weighted(x, theta)
        error -= beta * xw
        offset += num_params

    if yl is not None:
        ar_params = a[offset:]
        error -= np.dot(yl, ar_params)

    return error


def jacobian(a, x_list, y, yl, weight_methods):
    num_regressors = len(x_list)
    n_obs = len(y)

    offset = 1
    betas = a[offset:offset + num_regressors]
    offset += num_regressors

    intercept_jac = -1 * np.ones((n_obs, 1))
    beta_jac = []  # ∂e/∂β_i * xw
    weight_jac = []

    for i in range(num_regressors):
        x = x_list[i].values
        beta = betas[i]
        method = weight_methods[i]
        num_params = method.num_params
        theta = a[offset:offset + num_params]

        xw, _ = method.x_weighted(x, theta)
        jwx = jacobian_wx(x, theta, method)

        beta_jac.append(-xw.reshape(-1, 1))
        weight_jac.append(-beta * jwx)

        offset += num_params

    if yl is not None:
        ar_jac = -yl
        return np.hstack([intercept_jac] + beta_jac + weight_jac + [ar_jac])
    else:
        return np.hstack([intercept_jac] + beta_jac + weight_jac)


def jacobian_wx(x, params, weight_method):
    eps = 1e-6
    jt = []

    for i, p in enumerate(params):
        dp = np.concatenate([params[0:i], [p + eps / 2], params[i + 1:]])
        dm = np.concatenate([params[0:i], [p - eps / 2], params[i + 1:]])
        jtp, _ = weight_method.x_weighted(x, dp)
        jtm, _ = weight_method.x_weighted(x, dm)
        jt.append((jtp - jtm) / eps)

    return np.column_stack(jt)
