import numpy as np
from scipy.interpolate import interp1d
from lifelines import KaplanMeierFitter

class MeasureSurvLogloss:
    """
    Negative Log-Likelihood (Log-Loss) for survival predictions.

    Parameters
    ----------
    ERV : bool, optional (default=False)
        Whether to standardize measure against a Kaplan–Meier baseline
        (Explained Residual Variation).
    eps : float, optional (default=1e-6)
        Small numerical constant to avoid log(0).
    """

    def __init__(self, ERV=False, eps=1e-6):
        self.ERV = ERV
        self.eps = eps

    def _interp_pdf(self, surv_times, surv_probs, t):
        """
        Estimate predicted PDF at observed time `t` from survival curve S(t).
        PDF = -dS/dt
        """
        # Dérivée numérique de la survie (densité)
        pdf_vals = -np.gradient(surv_probs, surv_times)
        # Interpolation pour obtenir f(t)
        f_interp = interp1d(surv_times, pdf_vals, bounds_error=False, fill_value="extrapolate")
        return float(f_interp(t))

    def score(self, surv_funcs, event_times, event_observed):
        """
        Compute mean Negative Log-Likelihood.

        Parameters
        ----------
        surv_funcs : list of callable
            Each element is a function S_i(t) giving the survival probability at time t.
            (Ex: output of scikit-survival's `predict_survival_function`)
        event_times : array-like
            Observed event or censoring times.
        event_observed : array-like
            1 if event occurred, 0 if censored.

        Returns
        -------
        float : mean negative log-likelihood.
        """
        n = len(event_times)
        log_losses = []

        for i in range(n):
            S_i = surv_funcs[i]
            # grille temporelle et valeurs
            surv_times = np.array(S_i.x)
            surv_probs = np.array(S_i.y)
            t_i = event_times[i]

            f_ti = self._interp_pdf(surv_times, surv_probs, t_i)
            log_losses.append(-np.log(max(self.eps, f_ti)))

        nll = np.mean(log_losses)

        if self.ERV:
            # Comparaison avec Kaplan-Meier baseline
            kmf = KaplanMeierFitter().fit(event_times, event_observed)
            baseline_surv = kmf.survival_function_
            # On pourrait estimer une baseline log-loss et normaliser
            # (ici simplifié)
            nll_baseline = np.mean(-np.log(np.maximum(self.eps, baseline_surv.values.flatten())))
            explained_var = 1 - nll / nll_baseline
            return explained_var

        return nll
