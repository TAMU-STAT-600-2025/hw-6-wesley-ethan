// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do

// [[Rcpp::depends(RcppArmadillo)]]
//softmax helper
arma::mat softmax_c(const arma::mat& Z){
  int n = Z.n_rows;
  int K = Z.n_cols;
  arma::mat probs(n, K);
  
  for(int i = 0; i < n; i++) {
    arma::rowvec z_i = Z.row(i);
    arma::rowvec exp_z = exp(z_i);
    double sum_exp = arma::accu(exp_z);
    probs.row(i) = exp_z / sum_exp;
  }
  
  return probs;
}

// [[Rcpp::depends(RcppArmadillo)]]
//calc obj helper
double calc_obj_c(const arma::mat& X, const arma::uvec& y, 
                const arma::mat& beta, double lambda){
  int n = X.n_rows;
  
  // Compute probabilities
  arma::mat probs = softmax_c(X * beta);
  
  // Compute log-likelihood
  double ll = 0.0;
  for(int i = 0; i < n; i++) {
    ll += log(probs(i, y(i)));
  }
  
  // Compute ridge penalty
  double ridge_pen = (lambda / 2.0) * arma::accu(beta % beta);
  
  // Return objective: -ll + penalty
  return -ll + ridge_pen;
}

// [[Rcpp::depends(RcppArmadillo)]]
// For simplicity, no test data, only training data, and no error calculation.
// X - n x p data matrix
// y - n length vector of classes, from 0 to K-1
// numIter - number of iterations, default 50
// eta - damping parameter, default 0.1
// lambda - ridge parameter, default 1
// beta_init - p x K matrix of starting beta values (always supplied in right format)
// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                               int numIter = 50, double eta = 0.1, double lambda = 1){
    // All input is assumed to be correct
    
    // Initialize some parameters
    int K = y.max() + 1; // number of classes
    int p = X.n_cols;
    arma::mat beta = beta_init; // to store betas and be able to change them if needed
    arma::vec objective(numIter + 1); // to store objective values
    
    // Initialize anything else that you may need
    // Hessian init to a identity
    arma::mat I_p = arma::mat(p,p, arma::fill::eye);
    
    //obj at iter 0
    objective(0) = calc_obj_c(X, y, beta, lambda);
    
    // Newton's method cycle - implement the update EXACTLY numIter iterations
    for(int iter = 0; iter < numIter; iter++) {
      // get probs
      arma::mat probs = softmax_c(X * beta);
      
      for(int k = 0; k < K; ++k){
        arma::vec Pk = probs.col(k);
        
        //Generates the indicator of the Yk 
        arma::vec Yk = arma::zeros(y.n_elem);
        Yk.elem(arma::find(y == k)).ones();
        
        // weight matrix w "%" does elmentwise multiplication
        arma::vec w_k = Pk % (1.0 - Pk );
        
        // make XWX
        arma::mat XWX = X.t() * (X.each_col() % w_k);
        
        // Hessian
        arma::mat Hk = XWX + lambda * I_p;
        
        // Grad 
        arma::vec gk = X.t() * (Pk - Yk) + lambda * beta.col(k);
        
        // eta - eta * hess^-1 * grad
        arma::vec update = arma::solve(Hk, gk);
        beta.col(k) = beta.col(k) - eta * update;
      }
      
      objective(iter + 1) = calc_obj_c(X, y, beta, lambda);
    }
    
    // Create named list with betas and objective values
    return Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("objective") = objective);
}
