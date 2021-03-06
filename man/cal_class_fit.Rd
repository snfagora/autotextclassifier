% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.r
\name{cal_class_fit}
\alias{cal_class_fit}
\title{Evaluate a classification model output}
\usage{
cal_class_fit(model, test_x_class, test_y_class)
}
\arguments{
\item{model}{A classification model output}

\item{test_x_class}{Outcomes of the test dataset}

\item{test_y_class}{Predictors of the test dataset}
}
\value{
a dataframe of three columns \verb{(.metric, .estimator, .estimate)}
}
\description{
Evaluate a classification model output
}
