// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// Xtilde - centered and scaled X, n x p
// Ytilde - centered Y, n x 1
// lambda - tuning parameter
// beta0 - p vector of starting point for coordinate-descent algorithm, optional
// eps - precision level for convergence assessment, default 0.0001
// s - step size for proximal gradient
// [[Rcpp::export]]
arma::colvec fitLASSOstandardized_prox_Nesterov_c(const arma::mat& Xtilde, const arma::colvec& Ytilde,
                                                  double lambda, const arma::colvec& beta_start, 
                                                double eps = 0.0001, double s = 0.01){
  // All input is assumed to be correct
  
  // Initialize some parameters
  int n = Xtilde.n_rows, p = Xtilde.n_cols;
  arma::colvec beta(p);
  // ---- Implementation: Proximal Gradient with Nesterov Acceleration (t0 = 1) ----
  // Objective: (1/(2n)) * ||Ytilde - Xtilde * beta||^2 + lambda * ||beta||_1
  // Gradient of smooth part g(beta) = (1/(2n))||Y - Xb||^2 is:
  //   ∇g(beta) = (1/n) * X^T (X beta - Y)
  //
  // FISTA / Nesterov updates:
  //   t_{k+1} = (1 + sqrt(1 + 4 t_k^2))/2,  with t_0 = 1
  //   w_k     = (t_k - 1)/t_{k+1}
  //   y_k     = beta_k + w_k (beta_k - beta_{k-1})
  //
  // Prox step for L1 (soft-threshold):
  //   prox_{s*lambda ||.||_1}(z) = sign(z) * max(|z| - s*lambda, 0)
  
  arma::mat Xt = Xtilde.t();
  const double inv_n = 1.0 / static_cast<double>(n);
  
  arma::colvec beta_k = beta_start;       // current iterate beta_k
  arma::colvec beta_km1 = beta_start;     // previous iterate beta_{k-1}
  arma::colvec yk = beta_k;               // Nesterov point y_k
  
  double tk = 1.0;                        // initialize Nesterov coefficients at 1
  const int max_iter = 100000;            // hard guard to avoid infinite loop
  
  for (int it = 0; it < max_iter; ++it) {
    // Gradient at yk: (1/n) * X^T (X*yk - Y)
    arma::colvec grad = inv_n * (Xt * (Xtilde * yk - Ytilde));
    
    // Prox-gradient step
    arma::colvec z = yk - s * grad;
    
    // Soft-threshold: beta_new = sign(z) * max(|z| - s*lambda, 0)
    arma::colvec absz = arma::abs(z) - (s * lambda);
    absz.for_each( [](double& v){ if (v < 0.0) v = 0.0; } );
    arma::colvec beta_new = arma::sign(z) % absz;
    
    // Check convergence: relative parameter change
    double denom = std::max(1.0, arma::norm(beta_k, 2));
    double rel_change = arma::norm(beta_new - beta_k, 2) / denom;
    if (rel_change < eps) {
      beta = beta_new;
      break;
    }
    
    // Nesterov updates
    double tk1 = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * tk * tk));
    double w = (tk - 1.0) / tk1;
    arma::colvec yk_new = beta_new + w * (beta_new - beta_k);
    
    // Advance
    beta_km1 = beta_k;
    beta_k   = beta_new;
    yk       = yk_new;
    tk       = tk1;
    
    // If we reach last iteration without meeting eps, return last beta_new
    if (it == max_iter - 1) {
      beta = beta_new;
    }
  }
  return beta;
}
