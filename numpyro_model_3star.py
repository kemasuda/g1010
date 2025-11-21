__all__ = ["model_unresolved_multiple", "init_dict"]

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def smbound(x, low, upp, s=20, depth=30):
    """ sigmoid bound

        Args:
            x: parameter to be bounded
            low, upp: lower and upper bounds
            s: smoothness of the bound
            depth: depth of the bounds

        Returns:
            box-shaped penality term for the log-likelihood
            const if low < x < upp, const+depth otherwise

    """
    return -depth*(1./(jnp.exp(s*(x-low))+1)+1./(jnp.exp(-s*(x-upp))+1))


def sum_magnitudes(mags):
    return -2.5 * jnp.log10(jnp.sum(10**(-0.4 * mags)))


def flux_ratio_model(sf, params, alpha):
    feh = params['feh']
    rads = params['radius']
    teffs = params['teff']
    loggs = params['logg']
    flux = []
    for teff, logg, rad in zip(teffs, loggs, rads):
        flux.append(jnp.mean(sf.sm.sg.values(
            teff, logg, feh, alpha, sf.sm.wav_obs)) * rad**2)
    flux = jnp.array(flux)
    return flux / jnp.sum(flux)


def model_unresolved_multiple(self, sf, *, alpha=0., flux_ratio_obs=None, nstar=3, nodata=False, linear_age=True, flat_age_marginal=False, logamin=8, logamax=10.14,
                              fmin=-1, fmax=0.5, eepmin=0, eepmax=500, massmin=0.1, massmax=2.5, dist_scale=1.35, prot=None, prot_err=0.05, rho_obs=None, rho_err=None, age_obs=None, age_err=None):
    """model for NumPyro HMC

        Args:
            nodata: if True, data is ignored (i.e. sampling from prior)
            linear_age: if True/False, prior flat in age/logage is used
            flag_age_marginal: if True, the prior is set so that the marginal age/logage prior is flat.
                Otherwise, PDF has a constant value in the mass-(log)age-eep space.
            logamin, logmax: bounds for log10(age/yr)
            fmin, fmax: bounds for FeH
            eepmin, eepmax: bounds for EEP
            massmin, massmax: bounds for stellar mass
            dist_scale: length scale L in the distance prior (Bailer-Jones 2015; Astraatmadja & Bailer-Jones 2016)
            prot: if specified, gyrochronal log-likelihood is added folloing Angus et al. (2019), AJ 158, 173
            prot_err: error in Prot assumed in evaluating gyro log-likelihood
            rho, rho_err: if specified, gaussian prior on the mean density can be imposed (solar units)

    """
    ones = jnp.ones(nstar)

    # common parameters
    if linear_age:
        age = numpyro.sample("age", dist.Uniform(
            10**logamin/1e9, 10**logamax/1e9))
        # logage = jnp.log10(age * 1e9)
        # numpyro.deterministic("logage", logage)
        logage = numpyro.deterministic("logage", jnp.log10(age * 1e9))
    else:
        logage = numpyro.sample("logage", dist.Uniform(logamin, logamax))
        age = numpyro.deterministic("age", 10**logage/1e9)

    feh_init = numpyro.sample("feh_init", dist.Uniform(fmin, fmax))

    distance = numpyro.sample("distance", dist.Gamma(
        3, rate=1./dist_scale))  # BJ18, kpc
    parallax = numpyro.deterministic("parallax", 1. / distance)  # mas

    # star-by-star parameters
    eep = numpyro.sample("eep", dist.Uniform(
        eepmin*ones, eepmax*ones))  # length Nstar
    # each parameter have length len(eep)
    params = dict(zip(self.outkeys, self.mg.values(logage, feh_init, eep)))

    for key in self.outkeys:
        if 'mag' in key:
            params[key] = params[key] - 5 * \
                jnp.log10(parallax) + 10  # apparent mag
        params[key] = numpyro.deterministic(key, jnp.where(
            params[key] == params[key], params[key], -jnp.inf))  # remove nan
    # "feh" is set to be photospheric value
    params["feh"] = numpyro.deterministic(
        "feh", jnp.mean(params["feh_photosphere"]))
    params['parallax'] = parallax

    # likelihood
    if not nodata:
        for i, key in enumerate(self.obskeys):
            if 'mag' in key:
                mu = sum_magnitudes(params[key])
                numpyro.deterministic(f"{key}_total", mu)
            else:
                mu = params[key]
            numpyro.sample(f"obs_{key}", dist.Normal(
                mu, self.obserrs[i]), obs=self.obsvals[i])

    # mass prior
    logjac = jnp.log(params['dmdeep'])
    logjac += smbound(params['mass'], massmin, massmax)
    if flat_age_marginal:
        params['mmax'] = jnp.where(
            params['mmax'] < massmax, params['mmax'], massmax)
        logjac -= jnp.log(params['mmax'] - params['mmin'])

    logjac = jnp.where(logjac == logjac, logjac, -jnp.inf)
    numpyro.factor("logjac", logjac)

    if flux_ratio_obs is not None:
        for key in flux_ratio_obs.keys():
            flux_ratio = flux_ratio_model(sf, params, alpha)
            fratio_model = flux_ratio[1:]
            fratio_obs = flux_ratio_obs[key][0]
            fratio_err = flux_ratio_obs[key][1]
            numpyro.factor(
                f"fluxratio_{key}", -0.5 * jnp.sum((fratio_model - fratio_obs)**2 / fratio_err**2))
            for i in range(nstar):
                numpyro.deterministic(f"f{i+1}", flux_ratio[i])


def init_dict(self):
    dict = {}
    for key, val in zip(self.obskeys, self.obsvals):
        if key == "parallax":
            _val = np.mean(val)
            distkpc = 1. / _val if _val > 0 else 8.
            dict[key] = distkpc
        elif key == 'feh':
            dict['feh_init'] = np.mean(val)
        else:
            dict[key] = val

    return dict
