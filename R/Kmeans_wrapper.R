#' Title
#'
#' @param X 
#' @param K 
#' @param M 
#' @param numIter 
#'
#' @return Explain return
#' @export
#'
#' @examples
#' # Give example
MyKmeans <- function(X, K, M = NULL, numIter = 100){
  
  n = nrow(X) # number of rows in X
  
  # Check that X is a matrix, K and numIter are positive integers, M is either a matrix or NULL, and that K < n
  stopifnot(
    "X must be a numeric matrix" = is.numeric(X) && is.matrix(X),
    "K must be a positive integer" = is.numeric(K) && (K %% 1 == 0) && (K > 0),
    "numIter must be a positive integer" = is.numeric(K) && (numIter %% 1 == 0) && (numIter > 0),
    "M must be a numeric matrix or of value NULL" = any(is.numeric(M) && is.matrix(M), is.null(M)),
    "Number of rows in X must be greater than or equal to number of clusters, K" = (n >= K)
  )
  
  # Check whether M is NULL or not. If NULL, initialize based on K random points from X. If not NULL, check for compatibility with X dimensions.
  
  
  # Call C++ MyKmeans_c function to implement the algorithm
  Y = MyKmeans_c(X, K, M, numIter)
  
  # Return the class assignments
  return(Y)
}