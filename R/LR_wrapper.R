#' Title
#'
#' @param X 
#' @param y 
#' @param numIter 
#' @param eta 
#' @param lambda 
#' @param beta_init 
#'
#' @return
#' @export
#'
#' @examples
#' # Give example
LRMultiClass <- function(X, y, beta_init = NULL, numIter = 50, eta = 0.1, lambda = 1){
  
  # Compatibility checks from HW3 and initialization of beta_init
  n = nrow(X)
  p = ncol(X)
  K = length(unique(y))  # Num classes
  
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  if(!(all(X[, 1] == 1))){
    stop("First column of X not equal to 1's")
  }
  
  # Check for compatibility of dimensions between X and Y
  if(nrow(X) != length(y)){
    stop("Number of rows in X not equal to length of Y")
  }
  
  # Check for compatibility of dimensions between X and Xt
  if(ncol(X) != ncol(Xt)){
    stop("Number of cols in X not equal to number cols in Xt")
  }
  
  # Check eta is positive
  if(eta <= 0){
    stop("eta must be > 0")
  }
  
  # Check lambda is non-negative
  if(lambda < 0){
    stop("lambda must be >= 0")
  }
  
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  if(is.null(beta_init)){
    beta = matrix(0, nrow = p, ncol = K)
  } else { 
    # beta dim check
    if(nrow(beta_init) != p){
      stop(paste0("beta_init must have ", p, " rows"))
    }
    if(ncol(beta_init) != K){
      stop(paste0("beta_init must have ", K, " cols"))
    }
    
    beta = beta_init 
  }
  
  # Call C++ LRMultiClass_c function to implement the algorithm
  out = LRMultiClass_c(X, y, beta_init, numIter, eta, lambda)
  
  # Return the class assignments
  return(out)
}