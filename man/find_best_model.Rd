% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/classify_text.r
\name{find_best_model}
\alias{find_best_model}
\title{Find the best version of each algorithm based on the hyperparameters and 10-fold cross-validation.}
\usage{
find_best_model(
  lasso_wf,
  rand_wf,
  xg_wf,
  class_folds,
  lasso_grid,
  rand_grid,
  xg_grid,
  metric_choice = "accuracy"
)
}
\arguments{
\item{lasso_wf}{A lasso workflow (including the search space for the model)}

\item{rand_wf}{A random forest workflow (including the search space for the model)}

\item{xg_wf}{An XGBoost workflow (including the search space for the model)}

\item{class_folds}{10-fold cross-validation samples}

\item{lasso_grid}{The search spaces for lasso}

\item{rand_grid}{The search spaces for random forest}

\item{xg_grid}{The search space for XGBoost}

\item{metric_choice}{The selected metrics for the model evaluation among accuracy, balanced accuracy (bal_accuracy), F-score (f_means), and Area under the ROC curve (roc_auc). The default value is accuracy.}
}
\value{
A list output that contains the best model output for lasso, random forest, and XGBoost.
}
\description{
Find the best version of each algorithm based on the hyperparameters and 10-fold cross-validation.
}
