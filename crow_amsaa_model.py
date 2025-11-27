"""
Crow-AMSAA Reliability Growth Model Implementation

The Crow-AMSAA model is a continuous parameter model used to analyze failure data
over time. It models the cumulative number of failures as a power-law process.

Model: N(t) = λ * t^β

Where:
- N(t) is the cumulative number of failures by time t
- λ (lambda) is the scale parameter
- β (beta) is the shape parameter
  - β < 1: reliability growth (failures become less frequent)
  - β = 1: constant failure rate (exponential process)
  - β > 1: reliability decay (failures become more frequent)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
import pandas as pd
from typing import Tuple, Optional


class CrowAMSAA:
    """
    Crow-AMSAA reliability growth model for analyzing failure data.
    """
    
    def __init__(self):
        self.lambda_param = None
        self.beta_param = None
        self.fitted = False
        
    def fit(self, failure_times: np.ndarray, censoring_time: Optional[float] = None) -> Tuple[float, float]:
        """
        Fit the Crow-AMSAA model to failure data.
        
        Parameters:
        -----------
        failure_times : np.ndarray
            Array of failure times (cumulative time to each failure)
        censoring_time : float, optional
            Total observation time (if failures are right-censored)
            
        Returns:
        --------
        lambda_param : float
            Estimated scale parameter
        beta_param : float
            Estimated shape parameter
        """
        if len(failure_times) == 0:
            raise ValueError("No failure times provided")
        
        # Sort failure times
        failure_times = np.sort(failure_times)
        n = len(failure_times)
        
        # If no censoring time provided, use last failure time
        if censoring_time is None:
            censoring_time = failure_times[-1]
        
        # Maximum likelihood estimation
        # Log-likelihood function
        def neg_log_likelihood(params):
            lam, beta = params
            if lam <= 0 or beta <= 0:
                return 1e10
            
            # For complete data (no censoring)
            if censoring_time == failure_times[-1]:
                log_likelihood = n * np.log(lam * beta)
                log_likelihood += (beta - 1) * np.sum(np.log(failure_times))
                log_likelihood -= lam * np.sum(failure_times ** beta)
            else:
                # For censored data
                log_likelihood = n * np.log(lam * beta)
                log_likelihood += (beta - 1) * np.sum(np.log(failure_times))
                log_likelihood -= lam * censoring_time ** beta
            
            return -log_likelihood
        
        # Initial parameter estimates (method of moments approximation)
        # For initial beta, use relationship: beta ≈ n / sum(log(T/t_i))
        if n > 1:
            initial_beta = n / np.sum(np.log(censoring_time / failure_times))
        else:
            initial_beta = 1.0
        
        initial_lambda = n / (censoring_time ** initial_beta)
        
        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0=[initial_lambda, initial_beta],
            method='L-BFGS-B',
            bounds=[(1e-10, None), (1e-10, None)]
        )
        
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        
        self.lambda_param, self.beta_param = result.x
        self.fitted = True
        
        return self.lambda_param, self.beta_param
    
    def predict_cumulative_failures(self, time: float) -> float:
        """
        Predict cumulative number of failures at a given time.
        
        Parameters:
        -----------
        time : float
            Time point for prediction
            
        Returns:
        --------
        cumulative_failures : float
            Predicted cumulative number of failures
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.lambda_param * (time ** self.beta_param)
    
    def predict_failure_rate(self, time: float) -> float:
        """
        Predict failure rate (intensity function) at a given time.
        
        Parameters:
        -----------
        time : float
            Time point for prediction
            
        Returns:
        --------
        failure_rate : float
            Predicted failure rate (failures per unit time)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.lambda_param * self.beta_param * (time ** (self.beta_param - 1))
    
    def predict_next_failure_time(self, current_time: float) -> float:
        """
        Predict time to next failure from current time.
        
        Parameters:
        -----------
        current_time : float
            Current time point
            
        Returns:
        --------
        expected_time_to_failure : float
            Expected time to next failure
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Expected time to next failure from current time
        # This is an approximation for the power-law process
        if self.beta_param >= 1:
            # For beta >= 1, use mean of conditional distribution
            return ((current_time ** self.beta_param) + 1 / self.lambda_param) ** (1 / self.beta_param) - current_time
        else:
            # For beta < 1, use median approximation
            next_cumulative = self.predict_cumulative_failures(current_time) + 1
            next_time = (next_cumulative / self.lambda_param) ** (1 / self.beta_param)
            return next_time - current_time
    
    def get_parameters(self) -> Tuple[float, float]:
        """Get fitted parameters."""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        return self.lambda_param, self.beta_param
    
    def get_reliability_growth_indicator(self) -> str:
        """
        Interpret the beta parameter to determine reliability trend.
        
        Returns:
        --------
        indicator : str
            "Growth", "Constant", or "Decay"
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        if self.beta_param < 0.95:
            return "Growth"
        elif self.beta_param > 1.05:
            return "Decay"
        else:
            return "Constant"
    
    def calculate_confidence_intervals(self, failure_times: np.ndarray, 
                                      censoring_time: Optional[float] = None,
                                      confidence: float = 0.95) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculate confidence intervals for parameters using chi-square distribution.
        
        Parameters:
        -----------
        failure_times : np.ndarray
            Array of failure times
        censoring_time : float, optional
            Total observation time
        confidence : float
            Confidence level (default 0.95)
            
        Returns:
        --------
        lambda_ci : tuple
            Confidence interval for lambda
        beta_ci : tuple
            Confidence interval for beta
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        failure_times = np.sort(failure_times)
        n = len(failure_times)
        
        if censoring_time is None:
            censoring_time = failure_times[-1]
        
        # Chi-square critical values
        alpha = 1 - confidence
        chi2_lower = chi2.ppf(alpha / 2, df=2 * n)
        chi2_upper = chi2.ppf(1 - alpha / 2, df=2 * n)
        
        # Confidence intervals for beta
        # Using Fisher information matrix approximation
        # This is a simplified approach - full implementation would use
        # the covariance matrix from the Hessian
        
        # For beta, approximate CI using log transformation
        log_beta_std = 1.0 / np.sqrt(n)  # Simplified approximation
        z_score = 1.96  # For 95% CI
        
        beta_lower = self.beta_param * np.exp(-z_score * log_beta_std)
        beta_upper = self.beta_param * np.exp(z_score * log_beta_std)
        
        # For lambda, use relationship with beta
        lambda_lower = (2 * n) / (chi2_upper * (censoring_time ** beta_lower))
        lambda_upper = (2 * n) / (chi2_lower * (censoring_time ** beta_upper))
        
        return (lambda_lower, lambda_upper), (beta_lower, beta_upper)

