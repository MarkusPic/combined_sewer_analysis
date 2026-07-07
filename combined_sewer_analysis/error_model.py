import pandas as pd
import numpy as np

from combined_sewer_analysis import AnalyseData
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_process import arma_generate_sample


class ResidualUncertaintyModel:
    """
    Residual uncertainty model.

    The deterministic continuum model predicts the expected value, while this
    class simulates the remaining residual uncertainty around that prediction.

        residual_t = sigma_t * z_t

    sigma_t : time-varying residual standard deviation
    z_t     : standardized residual with temporal autocorrelation

    The model separates two effects:

    1. Heteroscedasticity:
       The residual variance may change over time. This is represented by
       sigma_t, which comes from the uncertainty estimated by AnalyseData.

    2. Autocorrelation:
       Residuals may remain high or low for several time steps. This temporal
       dependence is represented by an autoregressive model fitted to z_t.

    In sampling, the final simulated signal is constructed as

        continuum prediction + simulated residual
    """

    def __init__(self, ar_lags=1, min_sigma=1e-6, random_state=None):
        """
        Parameters
        ----------
        ar_lags : int
            Number of autoregressive lags used to model temporal dependence
            in the standardized residuals.

        min_sigma : float
            Lower bound for residual uncertainty. This is intended to prevent
            division by zero or unrealistically large standardized residuals
            when sigma_t is extremely small.

        random_state : int or None
            Random seed used to make Monte-Carlo simulations reproducible.
        """

        # Store model configuration.
        self.ar_lags = ar_lags
        self.min_sigma = min_sigma
        self.random_state = random_state

        # The AnalyseData object is stored after fitting because it provides
        # the continuum prediction and uncertainty series needed during sampling.
        self._csa = None

        # Empirical mean of the standardized residuals. The AR model is fitted
        # to centered residuals, and this mean is restored during simulation.
        self.z_mean_ = None

        # Fitted statsmodels AutoReg result.
        self.ar_model_ = None

        # Standard deviation of the AR model innovations. This represents the
        # remaining random variation after autoregressive structure is removed.
        self.innovation_std_ = None

    def fit(self, csa):
        """
        Train the residual uncertainty model from historical data.

        Residuals are standardized before fitting so the autoregressive model
        captures temporal dependence independently of the time-varying residual
        variance.

        Args:
            csa (AnalyseData): Calibrated analysis object containing the continuum prediction, residual uncertainty estimates, and observed residuals.

        Returns:
            ResidualUncertaintyModel: The fitted model instance.
        """
        # Keep a reference to the analysis object so the same continuum and
        # uncertainty series can be used later when generating simulations.
        self._csa = csa

        # Select the dry-weather periods used for calibration. The boolean mask
        # ensures that residual modelling is based only on the intended subset
        # of observations.
        dw_bool = csa.get_dw_bool_series(False)

        # Raw residuals are the difference between observed values and the
        # deterministic continuum prediction.
        residuals = csa.get_dw_residual_series(dw_bool)

        # Convert residuals into standardized residuals:
        #
        #     z_t = residual_t / sigma_t
        #
        # This removes the time-varying scale of the residuals, leaving a series
        # whose remaining structure should mainly be temporal dependence.
        sigma = csa.get_dw_uncertainty_series().loc[dw_bool].clip(lower=self.min_sigma)
        z = residuals / sigma

        # Remove the empirical mean before fitting the AR model. This makes the
        # autoregressive model describe deviations around zero rather than
        # mixing the mean level with the dependence structure.
        self.z_mean_ = z.mean()
        z_centered = z - self.z_mean_

        # Fit an autoregressive model to the centered standardized residuals.
        # The fitted coefficients describe how much previous residual states
        # influence the current residual state.
        self.ar_model_ = AutoReg(z_centered, lags=self.ar_lags, old_names=False).fit()

        # The residuals of the AR model are the innovations: the part of z_t
        # that cannot be explained by previous time steps. Their standard
        # deviation is used as the noise scale in simulations.
        self.innovation_std_ = self.ar_model_.resid.std()

        return self

    def _simulate_z(self, index: pd.DatetimeIndex, n_samples: int) -> pd.DataFrame:
        """

        Simulate standardized residual trajectories.

        The returned values are still dimensionless standardized residuals.
        They are later multiplied by sigma_t to obtain residuals in the
        original units.

        Args:
            index (pd.DatetimeIndex): Datetime index defining the simulation period.
            n_samples (int): Number of independent Monte-Carlo realizations.

        Returns:
            pd.DataFrame: Standardized residual simulations with timestamps as rows and simulation numbers as columns.
        """
        # Create a random number generator. Using the stored random_state makes
        # the simulations reproducible when a seed is provided.
        rng = np.random.default_rng(self.random_state)

        # Extract fitted AR parameters. statsmodels stores the intercept first,
        # followed by the autoregressive coefficients.
        params = self.ar_model_.params
        intercept = params.iloc[0]
        ar_coefs = params.iloc[1:].to_numpy()

        # Number of requested simulation time steps.
        n_steps = len(index)

        # Simulated AR processes need a burn-in period because their initial
        # values are artificial. Discarding early values allows the process to
        # approach its stationary behaviour before the requested period begins.
        burnin = max(100, 10 * self.ar_lags)

        # statsmodels' arma_generate_sample simulates general ARMA processes.
        # An AR(p) model is represented by the polynomial:
        #
        #     1 - phi_1 L - phi_2 L^2 - ... - phi_p L^p
        #
        # The moving-average part is set to 1 because no MA component is used.
        ar = np.r_[1.0, -ar_coefs]
        ma = np.array([1.0])

        def distrvs(size):
            """
            Draw AR innovations.

            These random shocks have the same standard deviation as the
            innovation residuals estimated during fitting.
            """
            return rng.normal(loc=0.0, scale=self.innovation_std_, size=size)

        # Generate n_samples independent standardized residual trajectories.
        # Each trajectory has n_steps plus burn-in values; the burn-in segment
        # is removed after simulation.
        simulations = arma_generate_sample(ar=ar, ma=ma, nsample=(n_samples, n_steps + burnin), distrvs=distrvs, axis=1, burnin=burnin)

        # Convert the fitted intercept to the unconditional mean of the AR
        # process. For a stationary AR model, the mean is:
        #
        #     intercept / (1 - sum(phi))
        #
        # The empirical z_mean_ is then added back because the AR model was
        # fitted to centered residuals.
        if abs(1.0 - ar_coefs.sum()) < 1e-8:
            # If the denominator is almost zero, the process is close to a unit
            # root. In that case, avoid numerical explosion and fall back to the
            # empirical mean.
            z_mean = self.z_mean_
        else:
            z_mean = self.z_mean_ + intercept / (1.0 - ar_coefs.sum())

        simulations += z_mean

        # Return simulations as a DataFrame indexed by the requested timestamps.
        # Rows are time steps and columns are simulation numbers.
        return pd.DataFrame(simulations.T[burnin:], index=index)

    def sample(self, index, n_samples=1000):
        """
        Generate stochastic realizations of the modelled signal.

        Each realization is computed as

            continuum prediction + sigma_t * simulated z_t

        Args:
            index (pd.DatetimeIndex):
                Datetime index for the simulation period.

            n_samples (int, optional):
                Number of Monte-Carlo realizations to generate.
                Defaults to 1000.

        Returns:
            pd.DataFrame: Simulated trajectories where rows correspond to timestamps and columns correspond to individual realizations.
        """
        # Simulate dimensionless standardized residuals with temporal
        # autocorrelation.
        z_sim = self._simulate_z(index=index, n_samples=n_samples)

        # Retrieve sigma_t for the requested period and enforce a lower bound
        # for numerical safety.
        sigma = self._csa.get_dw_uncertainty_series().loc[index].clip(lower=self.min_sigma)

        # Retrieve the deterministic continuum prediction for the same period.
        continuum = self._csa.get_dw_continuum_series().loc[index]

        # Transform standardized residuals back to original units by
        # multiplying with sigma_t, then add the deterministic continuum.
        return z_sim.mul(sigma, axis=0).add(continuum, axis=0)
