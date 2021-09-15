#' Evaluate a classification model output
#'
#' @param model A classification model output
#' @param test_y_class Predictors of the test dataset
#' @param test_x_class Outcomes of the test dataset
#' @return a dataframe of three columns `(.metric, .estimator, .estimate)`
#' @importFrom tidyr tibble
#' @importFrom yardstick metrics
#' @importFrom yardstick metric_set
#' @importFrom yardstick roc_auc
#' @importFrom yardstick accuracy
#' @importFrom yardstick bal_accuracy
#' @importFrom yardstick f_meas
#' @importFrom stats predict
#' @export
#'
cal_class_fit <- function(model, test_x_class, test_y_class) {

  # Evaluation
  metrics <- metric_set(accuracy, bal_accuracy, f_meas)

  out <- tibble(
    truth = test_y_class,
    predicted = predict(model, test_x_class)$.pred_class
  ) %>%
    metrics(truth = truth, estimate = predicted)

  return(out)
}

# The following visualization code draws on [Diego Usai's medium post](https://towardsdatascience.com/modelling-with-tidymodels-and-parsnip-bae2c01c131c).

#' Visualize a classification model output
#'
#' @param model A classification model output
#' @param model_title A classification model name
#' @param test_y_class Predictors of the test dataset
#' @param test_x_class Outcomes of the test dataset
#' @param metric_type A type of the metrics ("class" or "probability"). Either a "class"-based metrics (e.g., accuracy, balanced accuracy, F-score) or a "probability"-based metrics (e.g., ROC Curve). The default type is "class."
#' @return a bar or line plot(s)
#' @importFrom tidyr tibble
#' @importFrom dplyr bind_cols
#' @importFrom dplyr bind_rows
#' @importFrom yardstick metrics
#' @importFrom yardstick metric_set
#' @importFrom yardstick roc_auc
#' @importFrom yardstick accuracy
#' @importFrom yardstick bal_accuracy
#' @importFrom yardstick f_meas
#' @importFrom stats predict
#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 geom_text
#' @importFrom ggplot2 geom_col
#' @importFrom ggplot2 labs
#' @importFrom ggplot2 ylim
#' @importFrom ggplot2 autoplot
#' @importFrom glue glue
#' @export
#'

viz_class_fit <- function(model, model_title, test_x_class, test_y_class, metric_type = "class") {
  if (metric_type == "class") {
    metrics <- yardstick::metric_set(accuracy, bal_accuracy, f_meas)

    out <- tibble(
      truth = test_y_class,
      predicted = predict(model, test_x_class)$.pred_class
    ) %>%
      metrics(truth = truth, estimate = predicted) %>%
      ggplot(aes(x = glue("{toupper(.metric)}"), y = .estimate)) +
      geom_col() +
      labs(
        x = "Metrics",
        y = "Estimate"
      ) +
      ylim(c(0, 1)) +
      geom_text(aes(label = round(.estimate, 2)),
        color = "red"
      ) +
      labs(title = model_title)
  }

  if (metric_type == "probability") {
    metrics <- yardstick::metric_set(roc_auc)

    out <- tibble(
      truth = test_y_class,
      Class1 = predict(model, test_x_class, type = "prob")$.pred_FALSE.
    ) %>%
      roc_curve(truth, Class1) %>%
      autoplot()
  }

  return(out)
}

# The following function is adapted from https://juliasilge.com/blog/animal-crossing/

#' Visualize the importance of top 20 features
#
#' @param model model outputs
#' @return A bar plot
#' @importFrom dplyr top_n
#' @importFrom dplyr ungroup
#' @importFrom dplyr mutate
#' @importFrom stringr str_remove
#' @importFrom forcats fct_reorder
#' @importFrom ggplot2 aes
#' @importFrom ggplot2 geom_col
#' @importFrom ggplot2 theme
#' @importFrom ggplot2 labs
#' @export

topn_vip <- function(model) {
  model %>%
    top_n(20, wt = abs(Importance)) %>%
    ungroup() %>%
    mutate(
      Importance = abs(Importance),
      Variable = str_remove(Variable, "tfidf_text_"),
      # str_remove only removed one of the two `
      Variable = gsub("`", "", Variable),
      Variable = fct_reorder(Variable, Importance)
    ) %>%
    ggplot(aes(x = Importance, y = Variable)) +
    geom_col(show.legend = FALSE) +
    labs(y = NULL) +
    theme(text = element_text(size = 20))
}
