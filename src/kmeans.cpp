// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::uvec MyKmeans_c(const arma::mat& X, int K,
                            const arma::mat& M, int numIter = 100){
    // All input is assumed to be correct
    
    // Initialize some parameters
    int n = X.n_rows;
    int p = X.n_cols;
    arma::uvec Y(n); // to store cluster assignments
    
    // Initialize any additional parameters if needed
    arma::vec ones_n(n, arma::fill::ones);
    arma::vec ones_k(K, arma::fill::ones);
    arma::mat M_new = M;
    arma::mat M_old = M;
    int sum_unique = 0.5 * (K - 1) * K;
    
    // For loop with kmeans algorithm
    for (int i = 0; i < numIter; i++){
      // Calculate distance matrix
      arma::mat distance = (arma::sum(X % X, 1) * ones_k.t()) + (ones_n * arma::sum(M_new % M_new, 1).t()) - 2 * X * M_new.t();
      
      // Update each classification
      for (int r = 0; r < n; r++){
        Y(r) = distance.row(r).index_min();
      }
      
      // Stop if a cluster disappears
      if (arma::accu(arma::unique(Y)) < sum_unique) {
         Rcpp::stop("One of the clusters has disappeared, indicating a poor starting point. Give a new matrix, M, and try again.");
      }
      
      for (int cl = 0; cl < K; cl++) {
        // Update each cluster means
        int n_cl = arma::accu(Y == cl);
        M_new.row(cl) = arma::sum(X.rows(arma::find(Y == cl)))/n_cl;
      }
      
      if (arma::accu(M_old - M_new) == 0) {
        break;
      }
      
      M_old = M_new;
    }
    
    // Returns the vector of cluster assignments
    return(Y + 1);
}

