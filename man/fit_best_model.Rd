% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/classify_text.r
\name{fit_best_model}
\alias{fit_best_model}
\title{Fit the best model from each algorithm to the data.}
\usage{
fit_best_model(
  lasso_wf,
  best_lasso,
  rand_wf,
  best_rand,
  xg_wf,
  best_xg,
  train_x_class,
  train_y_class,
  category
)
}
\arguments{
\item{lasso_wf}{A lasso workflow (including the search space for the model)}

\item{best_lasso}{The best model output for lasso}

\item{rand_wf}{A random forest workflow (including the search space for the model)}

\item{best_rand}{The best model output for random forest}

\item{xg_wf}{An XGBoost workflow (including the search space for the model)}

\item{best_xg}{The best model output for XGBoost}

\item{train_x_class}{Training data for predictors}

\item{train_y_class}{Training data for outcomes}

\item{category}{The target binary category.}
}
\value{
A list output that contains the best output for lasso, random forest, and XGBoost.
}
\description{
Fit the best model from each algorithm to the data.
}
