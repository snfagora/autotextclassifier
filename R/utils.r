
# The following visualization code draws on [Diego Usai's medium post](https://towardsdatascience.com/modelling-with-tidymodels-and-parsnip-bae2c01c131c).

#' Visualize regression model outputs
#'
#' @param models regression model outputs
#' @param names regression model names
#' @return a point plot(s)
#' @importFrom tidyr tibble
#' @importFrom dplyr bind_cols
#' @importFrom parsnip predict.model_fit
#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 geom_abline
#' @importFrom ggplot2 geom_point
#' @importFrom ggplot2 labs
#' @importFrom glue glue
#' @importFrom tune coord_obs_pred
#' @export

viz_reg_fit <- function(models, names){

    # Bind ground truth and predicted values
    bind_cols(tibble(truth = test_y_reg), # Ground truth
              predict.model_fit(model, test_x_reg)) %>% # Predicted values

        # Visualize the residuals
        ggplot(aes(x = truth, y = .pred)) +
        # Diagonal line
        geom_abline(lty = 2) +
        geom_point(alpha = 0.5) +
        # Make X- and Y- scale uniform
        coord_obs_pred() +
        labs(title = glue::glue("{names}"))

}

#' Visualize classification model outputs
#'
#' @param models classification model outputs
#' @param names classification model names
#' @return a bar plot(s)
#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 geom_text
#' @importFrom ggplot2 geom_col
#' @importFrom ggplot2 labs
#' @importFrom ggplot2 ylim
#' @importFrom glue glue
#' @export
#'

viz_class_fit <- function(model){
    evaluate_class_fit(model) %>%
        ggplot(aes(x = glue("{toupper(.metric)}"), y = .estimate)) +
        geom_col() +
        labs(x = "Metrics",
             y = "Estimate") +
        ylim(c(0,1)) +
        geom_text(aes(label = round(.estimate, 2)),
                  size = 10,
                  color = "red")
}

#' Evaluate regression model outputs
#
#' @param model regression model outputs
#' @return A dataframe of two columns (truth, estimate)
#' @importFrom tidyr tibble
#' @importFrom dplyr bind_cols
#' @importFrom parsnip predict.model_fit
#' @importFrom yardstick metrics
#' @export

evaluate_reg_fit <- function(model){

    # Bind ground truth and predicted values
    bind_cols(tibble(truth = test_y_reg), # Ground truth
              predict.model_fit(model, test_x_reg)) %>% # Predicted values

        # Calculate root mean-squared error
        metrics(truth = truth, estimate = .pred)
}

#' Evaluate classification model outputs
#
#' @param model classification model outputs
#' @return A dataframe of two columns (truth, estimate)
#' @importFrom tidyr tibble
#' @importFrom dplyr bind_cols
#' @importFrom parsnip predict.model_fit
#' @importFrom yardstick metrics
#' @export

evaluate_class_fit <- function(model){

    # Bind ground truth and predicted values
    df <- bind_cols(tibble(truth = test_y_class), # Ground truth
                    predict.model_fit(model, test_x_class)) # Predicted values

    # Calculate metrics
    df %>% metrics(truth = truth, estimate = .pred_class)

}

# The following function is adapted from https://juliasilge.com/blog/animal-crossing/

#' Visualize the importance of top 20 features
#
#' @param df model outputs
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

topn_vip <- function(df) {
    df %>%
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
