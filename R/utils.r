
# The following visualization code draws on [Diego Usai's medium post](https://towardsdatascience.com/modelling-with-tidymodels-and-parsnip-bae2c01c131c).

#' Visualize a classification model output
#'
#' @param model A classification model output
#' @param model_title A classification model name
#' @param test_y_class Predictors of the test dataset
#' @param test_x_class Outcomes of the test dataset
#' @return a bar plot(s)
#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 geom_text
#' @importFrom ggplot2 geom_col
#' @importFrom ggplot2 labs
#' @importFrom ggplot2 ylim
#' @importFrom glue glue
#' @export
#'

viz_class_fit <- function(model, model_title, test_x_class, test_y_class){
    evaluate_class_fit(model, test_x_class, test_y_class) %>%
        ggplot(aes(x = glue("{toupper(.metric)}"), y = .estimate)) +
        geom_col() +
        labs(x = "Metrics",
             y = "Estimate") +
        ylim(c(0,1)) +
        geom_text(aes(label = round(.estimate, 2)),
                  size = 10,
                  color = "red") +
        labs(title = model_title)
}

#' Evaluate a classification model output
#
#' @param model A classification model output
#' @param test_y_class Predictors of the test dataset
#' @param test_x_class Outcomes of the test dataset
#' @return A dataframe of two columns (truth, estimate)
#' @importFrom tidyr tibble
#' @importFrom dplyr bind_cols
#' @importFrom yardstick metrics
#' @importFrom stats predict
#' @export

evaluate_class_fit <- function(model, test_x_class, test_y_class){

    # Bind ground truth and predicted values
    out <- bind_cols(tibble(truth = test_y_class), # Ground truth
                     predict(model, test_x_class)) # Predicted values

    # Calculate metrics
    out %>% metrics(truth = truth, estimate = .pred_class)

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
