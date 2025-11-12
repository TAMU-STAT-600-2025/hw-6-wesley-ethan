#' Title
#'
#' @param X 
#' @param Y
#' @param lambda 
#' @param beta_start 
#' @param eps 
#' @param s 
#'
#' @returns
#' @export
#'
#' @examples
fitLASSO_prox_Nesterov <- function(X, Y, lambda, 
                                   beta_start = NULL, eps = 0.0001, s = 0.01){
  
  # Compatibility checks from ProximalExamples and initialization of beta_init
  stopifnot(is.matrix(X), is.numeric(X), is.numeric(Y))
  n <- nrow(X); p <- ncol(X)
  stopifnot(length(Y) == n, length(lambda) == 1L, lambda >= 0)
  storage.mode(X) <- "double"
  Y <- as.numeric(Y)
  if (is.null(beta_start)) {
    beta_start <- rep(0, p)
  } else {
    stopifnot(length(beta_start) == p, is.numeric(beta_start))
    beta_start <- as.numeric(beta_start)
  }
  # Center and standardize X,Y as in HW4
  # Center Y; scale X columns so that (1/n) * t(X_j) %*% X_j = 1
  Ybar <- mean(Y)
  Ytilde <- Y - Ybar
  scale_vec <- sqrt(colSums(X^2) / n)
  scale_vec[scale_vec == 0] <- 1  # avoid divide-by-zero for zero columns
  Xtilde <- sweep(X, 2, scale_vec, "/")
  
  # Map starting beta from original scale to standardized scale:
  # theta = s ∘ beta  (so that Xtilde %*% theta = X %*% beta)
  beta_start_std <- beta_start * scale_vec
  
  # Call C++ fitLASSOstandardized_prox_Nesterov_c function to implement the algorithm
  beta_tilde = fitLASSOstandardized_prox_Nesterov_c(Xtilde, Ytilde, lambda, beta_start, eps, s)
  
  # Perform back scaling and centering to get original intercept and coefficient vector
  # Original-scale coefficients: beta = theta / s
  beta_coef <- as.numeric(beta_tilde) / scale_vec
  # Intercept on original scale: b0 = mean(Y) - colMeans(X) %*% beta
  X_means <- colMeans(X)
  b0 <- as.numeric(Ybar - sum(X_means * beta_coef))
  beta <- c(b0, beta_coef)
  
  # Return 
  # beta - the solution (without center or scale)
  # fmin - optimal function value (value of objective at beta, scalar)
  return(list(beta = beta, fmin = fmin))
}