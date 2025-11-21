import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import tinygp
from tinygp.kernels import quasisep as qk


def model_sb3(*, sf, param_bounds, rvshift2, rvshift3, keys_common=['alpha', 'feh'],
              f2_max=0.25, f3_max=0.25, empirical_vmacro=True, lnsigma_max=-3, single_wavres=False, lnc_max=2., fit_dilution=False, physical_logg_max=False, save_pred=False):
    """model for a single star

        Args:
            sf: SpecFit class
            param_bounds: dict of parameter bounds
            empirical_vmacro: if True, empirical vmacro-Teff relation is assumed
            single_wavres: if True, wavelength resolution is assumed to be common among orders
            fit_dilution: if True, f_dilution / f_star is fitted (assumed to be <1)
            physical_logg_max: if True, max of logg is determined as a function of Teff
            save_pred: if True, GP predictions are also saved

    """
    _sm = sf.sm
    par = {}
    ones3 = jnp.ones(3)

    for key in param_bounds.keys():
        if key == 'logg' and physical_logg_max:
            continue
        if key == 'zeta' and empirical_vmacro:
            continue
        if key == 'wavres' or key == 'rv':
            continue
        if key in keys_common:
            par[key+"_common_scaled"] = numpyro.sample(key+"_common_scaled", dist.Uniform(
                jnp.zeros_like(param_bounds[key][0]), jnp.ones_like(param_bounds[key][0])))
            par[key+"_common"] = numpyro.deterministic(
                key+"_common", par[key+"_common_scaled"] * (param_bounds[key][1] - param_bounds[key][0]) + param_bounds[key][0])
            par[key] = numpyro.deterministic(key, par[key+"_common"] * ones3)
        elif key == 'norm' or key == 'slope':
            par[key+"_scaled"] = numpyro.sample(key+"_scaled", dist.Uniform(
                jnp.zeros_like(param_bounds[key][0]), jnp.ones_like(param_bounds[key][1])))
            par[key] = numpyro.deterministic(
                key, par[key+"_scaled"] * (param_bounds[key][1] - param_bounds[key][0]) + param_bounds[key][0])
        else:
            par[key+"_scaled"] = numpyro.sample(key +
                                                "_scaled", dist.Uniform(ones3*0, ones3))
            par[key] = numpyro.deterministic(
                key, par[key+"_scaled"] * (param_bounds[key][1] - param_bounds[key][0]) + param_bounds[key][0])

    par['u1'] = numpyro.deterministic("u1", 2*jnp.sqrt(par["q1"])*par["q2"])
    par['u2'] = numpyro.deterministic("u2", jnp.sqrt(par["q1"])-par["u1"])

    if empirical_vmacro:
        par["zeta"] = numpyro.deterministic(
            "zeta", 3.98 + (par["teff"] - 5770.) / 650.)

    f2 = numpyro.sample("f2", dist.Uniform(0, f2_max))
    f3 = numpyro.sample("f3", dist.Uniform(0, f3_max))
    par['flux_ratio'] = numpyro.deterministic(
        "flux_ratio", jnp.array([f2, f3]))

    rv1 = numpyro.sample("rv1", dist.Uniform(
        param_bounds['rv'][0], param_bounds['rv'][1]))
    rv2 = numpyro.sample("rv2", dist.Uniform(
        param_bounds['rv'][0]+rvshift2, param_bounds['rv'][1]+rvshift2))
    rv3 = numpyro.sample("rv3", dist.Uniform(
        param_bounds['rv'][0]+rvshift3, param_bounds['rv'][1]+rvshift3))
    par['rv'] = numpyro.deterministic("rv", jnp.array([rv1, rv2, rv3]))

    ones = jnp.ones(_sm.Norder)
    # wavres_min = wavres_max
    if param_bounds['wavres'][0][0] == param_bounds['wavres'][1][0]:
        par['wavres'] = numpyro.deterministic(
            "wavres", param_bounds['wavres'][0])
    # wavres_min != wavres_max, order-independent wavres
    elif single_wavres:
        wavres_single = numpyro.sample("wavres", dist.Uniform(
            low=param_bounds['wavres'][0][0], high=param_bounds['wavres'][1][0]))
        par['wavres'] = ones * wavres_single
    # wavres_min != wavres_max, order-dependent wavres
    else:
        par['wavres'] = numpyro.sample("wavres", dist.Uniform(
            low=param_bounds['wavres'][0], high=param_bounds['wavres'][1]))

    fluxmodel = numpyro.deterministic(
        "fluxmodel", _sm.fluxmodel_multiorder(par))

    # GP log-likelihood
    lna = numpyro.sample("lna", dist.Uniform(low=-5, high=-0.5))
    lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=lnc_max))
    kernel = qk.Matern32(sigma=jnp.exp(lna), scale=jnp.exp(lnc))
    lnsigma = numpyro.sample(
        "lnsigma", dist.Uniform(low=-10, high=lnsigma_max))
    diags = _sm.error_obs**2 + jnp.exp(2*lnsigma)

    mask_all = _sm.mask_obs + (_sm.mask_fit > 0)
    idx = ~mask_all
    for j in range(len(fluxmodel)):
        idxj = idx[j]
        gp = tinygp.GaussianProcess(
            kernel, _sm.wav_obs[j][idxj], diag=diags[j][idxj], mean=0.0)
        flux_residual = numpyro.deterministic(
            "flux_residual%d" % j, _sm.flux_obs[j][idxj] - fluxmodel[j][idxj])
        numpyro.sample("obs%d" % j, gp.numpyro_dist(), obs=flux_residual)
        if save_pred:
            numpyro.deterministic("pred%d" % j, gp.predict(
                flux_residual, X_test=_sm.wav_obs[j]))
