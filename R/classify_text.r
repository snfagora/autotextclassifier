
#' Apply basic recipe to the recipe object. The basic recipe includes tokenization (using bigrams), removing stop words, filtering stop words by max tokens = 1,000, and normalization of document length using TF-IDF.
#'
#' @param recipe A recipe object. e.g., recipe(documents, output ~ predictors)
#' @param text The name of the text field in the data.
#' @return A prep object.
#' @importFrom recipes recipe
#' @importFrom recipes prep
#' @importFrom textrecipes step_tokenize
#' @importFrom textrecipes step_stopwords
#' @importFrom textrecipes step_tokenfilter
#' @importFrom textrecipes step_tfidf
#' @export

apply_basic_recipe <- function(recipe, text, token_ratio = 1000){

  recipe %>%
    # Used bigrams
    step_tokenize(text, token = "ngrams", options = list(n = 2)) %>%
    # Removed stopwords
    step_stopwords(text) %>%
    # Filtered tokens
    step_tokenfilter(text, max_tokens = token_ratio) %>%
    # Normalized document length
    step_tfidf(text) %>%
    prep()

}

#' Split data using stratified random sampling (SRS).
#'
#' @param data The data to be trained and tested.
#' @param category The target binary category.
#' @param prop_ratio The ratio used to split the data. The default value is 0.8
#' @return A list output that contains train_x_class, test_x_class, train_y_class, test_y_class.
#' @importFrom rsample initial_split
#' @importFrom dplyr mutate
#' @importFrom rsample training
#' @importFrom rsample testing
#' @importFrom recipes bake
#' @importFrom recipes all_predictors
#' @importFrom recipes all_outcomes
#' @export

split_using_srs <- function(data, category, prop_ratio = 0.8) {

  message("If you haven't done, please use set.seed() before running this function. It helps make the data splitting process reproducible.")

  # Split by stratified random sampling
  split_class <- initial_split(data,
                               mutate(category = as.factor(category)),
                               strata = category,
                               prop = prop_ratio)

  # training set
  raw_train_class <- training(split_class)
  raw_test_class <- testing(split_class)

  # x features (predictors)
  train_x_class <- bake(rec_articles,
                        raw_train_class, all_predictors())
  test_x_class <- bake(rec_articles,
                       raw_test_class, all_predictors())

  # y outcomes (outcomes)
  train_y_class <- bake(rec_articles,
                        raw_train_class, all_outcomes())$category %>% as.factor()

  test_y_class <- bake(rec_articles, raw_test_class, all_outcomes())$category %>% as.factor()

  # Putting together
  out <- list(train_x_class, test_x_class,
              train_y_class, test_y_class)

  return(out)

}

#' Create tuning parameters for algorithms (i.e., lasso, random forest, and XGBoost).
#'
#' @param mode The mode for the model specification.
#' @return A list output that contains the tuning parameters for lasso, random forest, and XGBoost.
#' @importFrom rsample initial_split
#' @importFrom parsnip logistic_reg
#' @importFrom parsnip set_engine
#' @importFrom parsnip set_mode
#' @importFrom parsnip rand_forest
#' @importFrom parsnip boost_tree
#' @importFrom tune tune
#' @export

create_tunes <- function(mode = "classification") {

  # Lasso spec
  lasso_spec <- logistic_reg(penalty = tune(), # tuning hyperparameter
                             mixture = 1) %>% # 1 = lasso, 0 = ridge
    set_engine("glmnet") %>%
    set_mode(mode)

  # Random forest spec
  rand_spec <-
    rand_forest(
      mode = mode,

      # Tuning hyperparameters
      mtry = tune(),
      min_n = tune()) %>%
    set_engine("ranger",
               seed = 1234,
               importance = "permutation")

  # XGBoost spec
  xg_spec <- boost_tree(

    # Mode
    mode = mode,

    # Tuning hyperparameters

    # The number of trees to fit, aka boosting iterations
    trees = tune(),
    # The depth of the decision tree (how many levels of splits).
    tree_depth = tune(),
    # Learning rate: lower means the ensemble will adapt more slowly.
    learn_rate = tune(),
    # Stop splitting a tree if we only have this many obs in a tree node.
    min_n = tune(),
    loss_reduction = tune(),
    # The number of randomly selected hyperparameters
    mtry = tune(),
    # The size of the data set used for modeling within an iteration
    sample_size = tune()
  ) %>%
    set_engine("xgboost")

  out <- list(lasso_spec, rand_spec, xg_spec)

  return(out)

}

#' Create search spaces for the algorithms based on the hyperparameters
#'
#' @param train_x_class Training data for predictors
#' @param category The target binary category.
#' @param lasso_spec The tuning parameters for lasso
#' @param rand_spec The tuning parameters for random forest
#' @param xg_spec The tuning parameters for XGBoost
#' @return A list of the workflows that include the search spaces for lasso, random forest, and XGBoost.
#' @importFrom dials grid_regular
#' @importFrom dials penalty
#' @importFrom dials mtry
#' @importFrom dials min_n
#' @importFrom dials grid_latin_hypercube
#' @importFrom dials trees
#' @importFrom dials tree_depth
#' @importFrom dials learn_rate
#' @importFrom dials loss_reduction
#' @importFrom dials sample_prop
#' @importFrom dials finalize
#' @importFrom workflows workflow
#' @importFrom workflows add_model
#' @importFrom workflows add_formula
#' @importFrom workflows update_model
#' @export

create_search_spaces <- function(train_x_class, category, lasso_sepc, rand_spec, xg_spec) {

  lambda_grid <- grid_regular(penalty(), levels = 50)

  rand_grid <- grid_regular(mtry(range = c(1, 10)),
                            min_n(range = c(2, 10)),
                            levels = 5)

  xg_grid <- grid_latin_hypercube(
    trees(),
    tree_depth(),
    learn_rate(),
    min_n(),
    loss_reduction(),
    sample_size = sample_prop(),
    finalize(mtry(), train_x_class),
    size = 30
  )

  # Lasso
  lasso_wf <- workflow() %>%
    add_model(lasso_spec) %>%
    add_formula(category ~ . )

  # Random forest
  rand_wf <- lasso_wf %>%
    update_model(rand_spec)

  # XGBoost
  xg_wf <- lasso_wf %>%
    update_model(xg_spec)

  out <- list(lasso_wf, rand_wf, xg_wf)

  return(out)

}

#' Create 10-fold cross-validation samples
#'
#' @param train_x_class Training data for predictors
#' @param train_y_class Training data for outcomes
#' @param category The target binary category.
#' @return 10-fold cross-validation samples
#' @importFrom rsample vfold_cv
#' @importFrom dplyr bind_cols
#' @importFrom tidyr tibble
#'

create_cv_folds <- function(train_x_class, train_y_class, category){

  # 10-fold cross-validation

  message("If you haven't done, please use set.seed() before running this function. It helps make the data splitting process reproducible.")

  class_folds <- vfold_cv(train_x_class %>% bind_cols(tibble(category = train_y_class)),
                        strata = category)

  return(class_folds)

}

#' Select the best output for each algorithm based on the hyperparameters and 10-fold cross-validation.
#'
#' @param lasso_wf A lasso workflow (including the search space for the model)
#' @param rand_wf A random forest workflow (including the search space for the model)
#' @param xg_wf An XGBoost workflow (including the search space for the model)
#' @return A list output that contains the best model output for lasso, random forest, and XGBoost.
# tune tune_grid
#' @importFrom tune control_grid
#' @importFrom tune select_best
#' @importFrom yardstick metric_set
#' @export

select_best <- function(lasso_wf, rand_wf, xg_wf){

  metrics <- metric_set(accuracy, precision, recall, f_meas)

  # Lasso
  lasso_res <- lasso_wf %>%
    tune_grid(
      resamples = class_folds,
      grid = lambda_grid,
      metrics = metrics
    )

  # Random forest
  rand_res <- rand_wf %>%
    tune_grid(
      resamples = class_folds,
      grid = rand_grid,
      control = control_grid(save_pred = TRUE),
      metrics = metrics
    )

  # XGBoost
  xg_res <- xg_wf %>%
    tune_grid(
      resamples = class_folds,
      grid = xg_grid,
      control = control_grid(save_pred = TRUE),
      metrics = metrics
    )

  best_lasso <- select_best(lasso_res, metric = "accuracy")
  best_rand <- select_best(rand_res, metric = "accuracy")
  best_xg <- select_best(xg_res, metric = "accuracy")

  out <- list(best_lasso, best_rand, best_xg)

  return(out)
}


#' Fit the best model from each algorithm to the data.
#'
#' @param lasso_wf A lasso workflow (including the search space for the model)
#' @param best_lasso The best model output for lasso
#' @param rand_wf A random forest workflow (including the search space for the model)
#' @param best_rand The best model output for random forest
#' @param xg_wf An XGBoost workflow (including the search space for the model)
#' @param best_xg The best model output for XGBoost
#' @param train_x_class Training data for predictors
#' @param train_y_class Training data for outcomes
#' @param category The target binary category.
#' @return A list output that contains the best output for lasso, random forest, and XGBoost.
# tune tune_grid
#' @importFrom tidyr tibble
#' @importFrom dplyr bind_cols
#' @importFrom tune finalize_workflow
#' @importFrom parsnip fit
#' @export

fit_best_models <- function(lasso_wf, best_lasso,
                            rand_wf, best_rand,
                            xg_wf, best_xg,
                            train_x_class,
                            train_y_class,
                            category) {
  # Lasso
  lasso_wf <- lasso_wf %>% finalize_workflow(best_lasso)

  # Random forest
  rand_wf <- rand_wf %>% finalize_workflow(best_rand)

  # XGBoost
  xg_wf <- xg_wf %>% finalize_workflow(best_xg)

  # Lasso
  lasso_fit <- lasso_wf %>%
    fit(train_x_class %>% bind_cols(tibble(category = train_y_class)))

  # Random forest
  rand_fit <- rand_wf %>%
    fit(train_x_class %>% bind_cols(tibble(category = train_y_class)))

  # XGBoost
  xg_fit <- xg_wf %>%
    fit(train_x_class %>% bind_cols(tibble(category = train_y_class)))

  out <- list(lasso_fit, rand_fit, xg_fit)

  return(out)

}
