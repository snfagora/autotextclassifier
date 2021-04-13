## code to prepare `DATASET` dataset goes here

# Load libraries
library(here)
library(tidyverse)

# Load data
sample_data <- load(file = here("inst/extdata/sample_data.rda"))

# Rename columns
names(sample_data) <- c("category", "org_name", "ein", "text")

# Overwrite data
usethis::use_data(sample_data, internal = TRUE, overwrite = TRUE)
