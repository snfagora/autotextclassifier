
#' Apply basic recipe to the dataframe that includes a text column. The basic recipe includes tokenization (using bigrams), removing stop words, filtering stop words by max tokens = 1,000, and normalization of document length using TF-IDF.
#'
#' @param input_data An input data.
#' @param formula A formula that specifies the relationship between the outcome and predictor variables (e.g, \code{category} ~ \code{text}.
#' @param text The name of the text column in the data.
#' @param remove_sparse_terms Remove sparse terms. The default value is TRUE.
#' @param token_threshold The maximum number of the tokens will be used in the classification.
#' @param add_embedding Add word embedding for feature engineering. The default value is NULL. Replace NULL with TRUE, if you want to add word embedding.
#' @param embed_dims Word embedding dimensions. The default value is 100.
#' @return A prep object.
#' @importFrom glue glue
#' @importFrom purrr map
#' @importFrom recipes recipe
#' @importFrom recipes prep
#' @importFrom recipes all_predictors
#' @importFrom textdata embedding_glove6b
#' @importFrom textrecipes step_tokenize
#' @importFrom textrecipes step_stopwords
#' @importFrom textrecipes step_tokenfilter
#' @importFrom textrecipes step_word_embeddings
#' @importFrom textrecipes step_tfidf
#' @export

apply_basic_recipe <- function(input_data, formula, text, token_threshold = 1000, add_embedding = NULL, embed_dims = 100){

  if (sum(is.na(input_data$text)) != 0) {

    warning("The text field includes missing values.")

  }

  if (sum(map(input_data$text, nchar) < 5) != 0) {

    warning("The text field includes very short documents (less than 5 words).")

  }

  message("Checked the missing values and extremely short documents.")

  rec_obj <- recipe(formula, input_data)

  message("Created the recipe object.")

  if (is.null(add_embedding)) {

    out <- rec_obj %>%
      # Used bigrams
      step_tokenize(text, token = "ngrams", options = list(n = 2)) %>%
      # Removed stopwords
      step_stopwords(text) %>%
      # Filtered tokens
      step_tokenfilter(text, max_tokens = token_threshold) %>%
      # Normalized document length
      step_tfidf(text) %>%
      prep()

      message(glue("Tokenized, removed stopd words, filtered up to the max_tokens = {token_threshold}, and normalized the document length using TF-IDF."))

  }

  if (!is.null(add_embedding)) {

    # Define the dimensions of word vectors
    glove6b <- textdata::embedding_glove6b(dimensions = embed_dims)

    out <- rec_obj %>%
      # Tokenize
      step_tokenize(text, options = list(strip_punct = FALSE)) %>%
      # Removed stopwords
      step_stopwords(text) %>%
      # Filtered tokens
      step_tokenfilter(text, max_tokens = 1000) %>%
      # Add word embedding
      step_word_embeddings(text, embeddings = glove6b) %>%
      prep()

      message(glue("Tokenized, removed stopd words, filtered up to the max_tokens = {token_threshold}, and added word embedding for feature engineering."))

  }

  return(out)

}

#' Creating training and testing data based on stratified random sampling (SRS) and preprocessing steps
#'
#' @param input_data The data to be trained and tested.
#' @param category The target binary category.
#' @param rec The recipe (preprocessing steps) that will be applied to the training and test data
#' @param prop_ratio The ratio used to split the data. The default value is 0.8
#' @param pull_id The identifier used to identify training and test data values. The default value is `NULL.`
#' @return A list output that contains train_x_class, test_x_class, train_y_class, test_y_class.
#' @importFrom rsample initial_split
#' @importFrom dplyr mutate
#' @importFrom dplyr pull
#' @importFrom rsample training
#' @importFrom rsample testing
#' @importFrom recipes bake
#' @importFrom recipes all_predictors
#' @importFrom recipes all_outcomes
#' @export

split_using_srs <- function(input_data, category, rec, prop_ratio = 0.8, pull_id = NULL) {

  message("If you haven't done, please use set.seed() before running this function. It helps make the data splitting process reproducible.")

  input_data <- input_data %>%
    mutate(category = as.factor(category))

  # Split by stratified random sampling
  split_class <- initial_split(input_data,
                               strata = category,
                               prop = prop_ratio)

  # training set
  raw_train_class <- training(split_class)
  raw_test_class <- testing(split_class)

  if (!is.null(pull_id)) {

    train_id <- raw_train_class %>%
      pull({{pull_id}})

    test_id <- raw_test_class %>%
      pull({{pull_id}})

  }

  # x features (predictors)
  train_x_class <- bake(rec,
                        raw_train_class, all_predictors())
  test_x_class <- bake(rec,
                       raw_test_class, all_predictors())

  # y outcomes (outcomes)
  train_y_class <- bake(rec,
                        raw_train_class, all_outcomes())$category

  test_y_class <- bake(rec, raw_test_class, all_outcomes())$category

  # Putting together
  if (is.null(pull_id)) {

  out <- list("train_x_class" = train_x_class,
              "test_x_class" = test_x_class,
              "train_y_class" = train_y_class,
              "test_y_class" = test_y_class) } else {

  out <- list("train_x_class" = train_x_class,
              "test_x_class" = test_x_class,
              "train_y_class" = train_y_class,
              "test_y_class" = test_y_class,
              "train_id" = train_id,
              "test_id" = test_id)

              }

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

  out <- list("lasso_spec" = lasso_spec,
              "rand_spec" = rand_spec,
              "xg_spec" = xg_spec)

  return(out)

}

#' Create search spaces for the algorithms based on the hyperparameters
#'
#' @param train_x_class Training data for predictors
#' @param category The target binary category
#' @param lasso_spec The tuning parameters for lasso
#' @param rand_spec The tuning parameters for random forest
#' @param xg_spec The tuning parameters for XGBoost
#' @return A list of the search spaces for lasso, random forest, and XGBoost.
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
#' @export

create_search_spaces <- function(train_x_class, category, lasso_spec, rand_spec, xg_spec) {

  lasso_grid <- grid_regular(penalty(), levels = 50)

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

  out <- list("lasso_grid" = lasso_grid,
              "rand_grid" = rand_grid,
              "xg_grid" = xg_grid)

  return(out)
}

#' Create workflows for the algorithms based on the hyperparameters
#'
#' @param lasso_spec The model specification for lasso
#' @param rand_spec The model specification for random forest
#' @param xg_spec The model specification for XGBoost
#' @param category The target binary category
#' @return A list of the workflows for lasso, random forest, and XGBoost.
#' @importFrom workflows workflow
#' @importFrom workflows add_model
#' @importFrom workflows add_formula
#' @importFrom workflows update_model
#' @export

create_workflows <- function(lasso_spec, rand_spec, xg_spec, category) {
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

  out <- list("lasso_wf" = lasso_wf,
              "rand_wf" = rand_wf,
              "xg_wf" = xg_wf)

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
#' @export

create_cv_folds <- function(train_x_class, train_y_class, category){

  # 10-fold cross-validation

  message("If you haven't done, please use set.seed() before running this function. It helps make the data splitting process reproducible.")

  class_folds <- vfold_cv(train_x_class %>%
                            bind_cols(tibble(category = train_y_class)),
                        strata = category)

  return(class_folds)

}

#' Find the best version of each algorithm based on the hyperparameters and 10-fold cross-validation.
#'
#' @param lasso_wf A lasso workflow (including the search space for the model)
#' @param rand_wf A random forest workflow (including the search space for the model)
#' @param xg_wf An XGBoost workflow (including the search space for the model)
#' @param class_folds 10-fold cross-validation samples
#' @param lasso_grid The search spaces for lasso
#' @param rand_grid The search spaces for random forest
#' @param xg_grid The search space for XGBoost
#' @param metric_choice The selected metrics for the model evaluation among accuracy, balanced accuracy (bal_accuracy), F-score (f_means), and Area under the ROC curve (roc_auc). The default value is accuracy.
#' @return A list output that contains the best model output for lasso, random forest, and XGBoost.
#' @importFrom tune tune_grid
#' @importFrom tune control_grid
#' @importFrom tune select_best
#' @importFrom yardstick metric_set
#' @importFrom yardstick accuracy
#' @importFrom yardstick bal_accuracy
#' @importFrom yardstick f_meas
#' @importFrom yardstick roc_auc
#' @export

find_best_model <- function(lasso_wf, rand_wf, xg_wf,
                        class_folds, lasso_grid, rand_grid, xg_grid,
                        metric_choice = "accuracy"){

  metrics <- metric_set(accuracy, bal_accuracy, f_meas, roc_auc)

  # Lasso
  lasso_res <- lasso_wf %>%
    tune_grid(
      resamples = class_folds,
      grid = lasso_grid,
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

  best_lasso <- select_best(lasso_res, metric = metric_choice)
  best_rand <- select_best(rand_res, metric = metric_choice)
  best_xg <- select_best(xg_res, metric = metric_choice)

  out <- list("best_lasso" = best_lasso,
              "best_rand" = best_rand,
              "best_xg" = best_xg)

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

fit_best_model <- function(lasso_wf, best_lasso,
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

  out <- list("lasso_fit" = lasso_fit,
              "rand_fit" = rand_fit,
              "xg_fit" = xg_fit)

  return(out)

}

#' Build a pipeline from creating tuning parameters, search spaces, workflows, 10-fold cross-validation samples to finding the best model from lasso, random forest, XGBoost to fitting the best model from each algorithm to the data
#'
#' @param input_data The data to be trained and tested.
#' @param category The target binary category.
#' @param rec The recipe (preprocessing steps) that will be applied to the training and test data
#' @param prop_ratio The ratio used to split the data. The default value is 0.8
#' @param metric_choice The selected metrics for the model evaluation among accuracy, balanced accuracy (bal_accuracy), F-score (f_means), and Area under the ROC curve (roc_auc). The default value is accuracy.
#' @return A list output that contains the best output for lasso, random forest, and XGBoost.
#' @importFrom zeallot `%<-%`
#' @export

build_pipeline <- function(input_data, category, rec, prop_ratio = 0.8, metric_choice = "accuracy") {

  # Split data
  c(train_x_class, test_x_class, train_y_class, test_y_class) %<-% split_using_srs(input_data, category = category, rec = rec)

  # Export these objects to the global environment
  assign("train_x_class", train_x_class, envir = globalenv())
  assign("train_y_class", train_y_class, envir = globalenv())
  assign("test_x_class", test_x_class, envir = globalenv())
  assign("test_y_class", test_y_class, envir = globalenv())

  # Create tuning parameters
  c(lasso_spec, rand_spec, xg_spec) %<-% create_tunes()

  # Create search spaces
  c(lasso_grid, rand_grid, xg_grid) %<-% create_search_spaces(train_x_class, category, lasso_spec, rand_spec, xg_spec)

  # Create workflows
  c(lasso_wf, rand_wf, xg_wf) %<-% create_workflows(lasso_spec, rand_spec, xg_spec, category)

  # Create 10-fold cross-validation samples
  class_folds <- create_cv_folds(train_x_class, train_y_class, category)

  # Find the best model from each algorithm
  c(best_lasso, best_rand, best_xg) %<-% find_best_model(lasso_wf, rand_wf, xg_wf, class_folds, lasso_grid, rand_grid, xg_grid, metric_choice = "accuracy")

  # Fit the best model from each algorithm to the data
  c(lasso_fit, rand_fit, xg_fit) %<-% fit_best_model(lasso_wf, best_lasso, rand_wf, best_rand, xg_wf, best_xg, train_x_class, train_y_class, category)

  # Rename the output
  out <- list("lasso_fit" = lasso_fit,
              "rand_fit" = rand_fit,
              "xg_fit" = xg_fit)

  return(out)
}
