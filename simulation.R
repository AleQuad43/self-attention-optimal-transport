setwd("~/Desktop/Tesi/sim")

library(tensorflow)
library(keras3)
library(readr)
library(dplyr)
library(mvtnorm)
library(expm)
library(transport)

reticulate::py_run_string("
import warnings
warnings.filterwarnings('ignore')
")

# TRANSFOMER FUNCTIONS ----------------------------------------------------

source("functions Transformer.R")

# TRANSFORMER SIMULATION --------------------------------------------------

Transformer_simulation <- function(
    data_means,
    data_variances,
    pretrain_means,
    pretrain_variances,
    epochs = 200,
    dimensions = c(90, 20, 2),
    repetitions = 10,
    pretrain = FALSE,
    save_plots = TRUE,
    verbose_training = TRUE,
    show_plots = TRUE,
    transformer_blocks = 1,
    mlp_units = c(8),
    mlp_dropout = 0.4,
    transformer_head_size = 16,
    transformer_num_heads = 10,
    transformer_ff_dim = 32,
    transformer_dropout = 0.1,
    learning_rate = 1e-2,
    plot_subset = 0.1,
    patience = epochs,
    plots_location = NA
) {

  verbose_training <- if(verbose_training == TRUE) 1 else 0

    # Output tables
  create_results_df <- function(n_classes, repetitions) {
    base_cols <- list(
      gauss_dist = numeric(repetitions),
      Accuracy_pw = numeric(repetitions),
      Accuracy_iw = numeric(repetitions),
      Computational_Time = numeric(repetitions),
      Matching = numeric(repetitions),
      Wass_Distance = numeric(repetitions),
      Transformer_Distance = numeric(repetitions),
      Transformer_Cost = numeric(repetitions),
      Monge_gap = numeric(repetitions),
      Efficiency = numeric(repetitions),
      Optimality = numeric(repetitions),
      best_epoch = numeric(repetitions)
    )
    
    recall_cols <- setNames(
      as.list(rep(list(numeric(repetitions)), n_classes)),
      paste0("Recall_class_", 0:(n_classes-1))
    )
    
    as.data.frame(c(base_cols, recall_cols))
  }
  
  results <- create_results_df(n_classes = length(data_means), repetitions = repetitions)
  
  fit_transformer <- function(pretrained = FALSE) {
    model <- build_model(
      input_shape = dim(x_train)[-1],
      head_size = transformer_head_size,
      num_heads = transformer_num_heads,
      ff_dim = transformer_ff_dim,
      num_transformer_blocks = transformer_blocks,
      mlp_units = mlp_units,
      mlp_dropout = mlp_dropout,
      dropout = transformer_dropout,
      y_train = y_train)
    
    model %>% compile(
      loss = c("sparse_categorical_crossentropy", rep(NULL, 2 * transformer_blocks)),
      optimizer = optimizer_rmsprop(learning_rate = learning_rate),
      metrics = c(list("accuracy"), rep(list(NULL), 2 * transformer_blocks))
    )
    
    if (pretrain) {
      load_model_weights(model, filepath = "MLP.weights.h5", skip_mismatch = TRUE)
      freeze_weights(model, from = "MLP_layer_1")
    }
    
    projections <- vector("list", transformer_blocks)
    for (j in 1:transformer_blocks) projections[[j]][[1]] <- x_train_s
    
    proj_cb <- callback_lambda(
      on_epoch_end = function(epoch, logs) {
        preds <- model %>% predict(x_train_s, verbose = verbose_training)
        preds_r <- reticulate::py_to_r(preds)
        for (j in 1:transformer_blocks) {
          projections[[j]][[epoch + 1]] <<- preds_r[[j + transformer_blocks + 1]]
        }
      }
    )
    
    early_stop <- callback_early_stopping(patience = patience, restore_best_weights = TRUE)
    callbacks <- list(proj_cb, early_stop)
    
    training_time <- system.time({
      history <- model %>%
        fit(x_train, y_train_ts,
            batch_size = dim(x_test)[1],
            shuffle = FALSE,
            epochs = epochs,
            callbacks = callbacks,
            validation_data = list(x_test_s, y_test_ts),
            verbose = verbose_training)
    })
    
    list(model = model, time = training_time[3], history = history,
         proj = projections[[transformer_blocks]][1:early_stop$best_epoch],
         best_epoch = early_stop$best_epoch)
  }

  # Main loop
  for (i in 1:repetitions) {
    
    t0 <- Sys.time()
    
    seed <- i
    set_random_seed(seed)
    retry_counter <- 0
    
    # Generate data for this repetition
    data <- generator(dimensions = dimensions, means = data_means, variances = data_variances, plot = show_plots)
    
    x_train <- data$x_train
    y_train <- data$y_train
    x_test <- data$x_test
    y_test <- data$y_test
    
    mean_train <- apply(x_train, 3, mean)
    sd_train <- apply(x_train, 3, sd)
    
    x_train_s <- sweep(sweep(x_train, 3, mean_train, "-"), 3, sd_train, "/")
    x_test_s <- sweep(sweep(x_test, 3, mean_train, "-"), 3, sd_train, "/")
    
    y_train_ts <- matrix(y_train, nrow = dim(x_train)[1], ncol = dim(x_train)[2])
    y_test_ts <- matrix(y_test, nrow = dim(x_test)[1], ncol = dim(x_test)[2])
    
    # Save Gaussian distance
    results$gauss_dist[i] <- mean(combn(seq_along(data_means), 2, function(j) wasserstein_gaussian(data_means[[j[1]]], data_variances[[j[1]]], data_means[[j[2]]], data_variances[[j[2]]]), simplify = TRUE))
    
    best_epoch <- 0
    while (best_epoch<2) {
      fit <- fit_transformer(pretrained = TRUE)
      best_epoch <- fit$best_epoch
      retry_counter <- retry_counter + 1
    }
    
    results$Computational_Time[i] <- fit$time
    results$Accuracy_pw[i] <- fit$history$metrics$val_MLP_layer_sm_accuracy[fit$best_epoch]
    
    # Recall
    preds <- fit$model %>% predict(x_test_s)
    avg_preds <- apply(preds[[1]], c(1, 3), mean)   # shape: [samples, classes]
    pred_labels <- max.col(avg_preds) - 1
    
    cm <- table(y_test, pred_labels)
    recalls <- diag(cm) / rowSums(cm)
    results$Accuracy_iw[i] <- sum(diag(cm))/sum(cm)

    results[i, paste0("Recall_class_", 0:(length(recalls)-1))] <- recalls
    
    results$best_epoch[i] <- fit$best_epoch
    
    trans_dist <- compute_transformer_distance(fit$proj, y_train)
    results$Transformer_Distance[i] <- trans_dist$total_cost
    results$Transformer_Cost[i] <- compute_transformer_cost(fit$proj)
    results$Efficiency[i] <- results$Transformer_Distance[i] / results$Transformer_Cost[i]
    
    wass <- compute_wasserstein_distance(fit$proj, y_train, fast = TRUE)
    results$Wass_Distance[i] <- wass$total_wass
    results$Optimality[i] <- results$Wass_Distance[i] / results$Transformer_Distance[i]
    matches <- sum(sapply(wass$transport_plans, function(plan) sum(plan$from == plan$to)))
    results$Matching[i] <- matches / (dim(y_train_ts)[1] * dim(y_train_ts)[2])
    results$Monge_gap[i] <- results$Transformer_Distance[i] - results$Wass_Distance[i]
    
    if (save_plots) {
      if (pretrain){
      name_1 <- paste0(plots_location, "/transformer_mapping_", round(results$gauss_dist[i],2), "_", i, "_pretrained.jpg") 
      name_2 <- paste0(plots_location, "/transformer_paths_", round(results$gauss_dist[i],2), "_", i, "_pretrained.jpg")
      } else {
        name_1 <- paste0(plots_location, "/transformer_mapping_", round(results$gauss_dist[i],2), "_", i, ".jpg")
        name_2 <- paste0(plots_location, "/transformer_paths_", round(results$gauss_dist[i],2), "_", i, ".jpg")
        }
      ind <- sample_common_indices(y_train, subset = plot_subset)
      jpeg(name_1, width = 800, height = 800)
      plot_transformer_mapping(fit$proj, y_train, ind, name = paste0("Transformer mapping_", round(results$gauss_dist[i],2)))
      add_cost_table(fit$proj, y_train, corner = "bottomleft", cex = 1, box_col = "gray95", transformer_info = trans_dist, stepwise_cost = results$Transformer_Cost[i])
      dev.off()
      
      jpeg(name_2, width = 800, height = 800)
      plot_transformer_paths(fit$proj, y_train, ind, name = paste0("Transformer path_", round(results$gauss_dist[i],2)))
      add_cost_table(fit$proj, y_train, corner = "bottomleft", cex = 1, box_col = "gray95",
                     wasserstein_info = wass, transformer_info = trans_dist, stepwise_cost = results$Transformer_Cost[i])
      dev.off()
    }

    cat("Repetition", i, "gaussian dist:", results$gauss_dist[i], " retries:", retry_counter-1, "time:", difftime(Sys.time(), t0, units = "secs"),  "\n")

    print(paste("Completed repetition", i))
  }
  
  return(list(results = results, retries = retry_counter))
}


# ... ---------------------------------------------------------------------

all_results <- list()

for (i in 1:4) {

  # means <- list(c(0, -i), c(0, i))
  # variances <- list(diag(2, 2), diag(2, 2))
  # cat("\n--- Running simulation with means: (", round(means[[1]],2), "), (", round(means[[2]],2), ")\n")
  
  x <- 6*i/((1+sqrt(2))*2)
  means <- list(c(-x, 0), c(x, 0), c(0, x))
  variances <- list(diag(2, 2), diag(2, 2), diag(2, 2))
  cat("\n--- Running simulation with means: (", round(means[[1]],2), "), (", round(means[[2]],2), ") and (", round(means[[3]],2), ")\n")
  
  # Run the simulation study
  result <- Transformer_simulation(
    data_means = means,
    data_variances = variances,
    pretrain_means = list(c(-5, 0), c(5, 0), c(0, 5)),
    pretrain_variances = list(diag(1, 2), diag(1, 2), diag(1, 2)),
    transformer_blocks = 2,
    repetitions = 100,
    pretrain = TRUE,
    save_plots = FALSE,
    show_plots = FALSE,
    epochs = 200,
    verbose_training = FALSE,
    # plots_location = "plots 2 nuvole easy"
  )
  
  # Store in a list
  all_results[[paste0("dist_", i)]] <- result
}

# all_results$dist_1$results$Transformer_Cost/all_results$dist_1$results$Wass_Distance
# all_results$dist_3$results_pretrained
# zapsmall(colMeans(all_results$dist_1$results)); zapsmall(apply(all_results$dist_1$results, 2, sd))
# zapsmall(colMeans(all_results$dist_1$results_pretrained)); zapsmall(apply(all_results$dist_1$results_pretrained, 2, sd))




# STORE RESULTS -----------------------------------------------------------

# Store essential parameters only
params <- list(
  repetitions = 100,
  dimensions = c(90, 20, 2),
  pretrain_means = list(c(-5, 0), c(5, 0), c(0, 5)),
  pretrain_variances = list(diag(1, 2), diag(1, 2), diag(1, 2)),
  transformer_blocks = 2,
  transformer_head_size = 16,
  transformer_num_heads = 10,
  transformer_ff_dim = 32,
  transformer_dropout = 0.1,
  mlp_units = c(8),
  mlp_dropout = 0.4,
  epochs = 200,
  learning_rate = 1e-2
)

# Recreate data configurations used
data_configurations <- list(
  dist_1 = list(
    means = list(c(-6/((1+sqrt(2))*2), 0),c(6/((1+sqrt(2))*2), 0), c(0, 6/((1+sqrt(2))*2))),
    variances = list(diag(2,2), diag(2,2), diag(2,2))
  ),
  dist_2 = list(
    means = list(c(-6*2/((1+sqrt(2))*2), 0),c(6*2/((1+sqrt(2))*2), 0), c(0, 6*2/((1+sqrt(2))*2))),
    variances = list(diag(2,2), diag(2,2), diag(2,2))
  ),
  dist_3 = list(
    means = list(c(-6*3/((1+sqrt(2))*2), 0),c(6*3/((1+sqrt(2))*2), 0), c(0, 6*3/((1+sqrt(2))*2))),
    variances = list(diag(2,2), diag(2,2), diag(2,2))
  ),
  dist_4 = list(
    means = list(c(-6*4/((1+sqrt(2))*2), 0),c(6*4/((1+sqrt(2))*2), 0), c(0, 6*4/((1+sqrt(2))*2))),
    variances = list(diag(2,2), diag(2,2), diag(2,2))
  )
)

# Save everything
saveRDS(
  list(
    results = all_results,
    parameters = params,
    data_configurations = data_configurations,
    session_info = sessionInfo()
  ),
  file = "results100/results_3_nuvole_pretrained_0°_summary"
)


# OT MODEL FUNCTIONS -----------------------------------------------------------

source("functions OT model.R")

# OT MODEL SIMULATION ----------------------------------------------------------

OT_model_simulation_new <- function(
    data_means,
    data_variances,
    epochs = 200,
    dimensions = c(90, 20, 2),
    repetitions = 10,
    
    # Training options
    mlp_units = c(32),
    mlp_dropout = 0.4,
    learning_rate = 1e-2,
    patience = epochs,
    verbose_training = TRUE,
    
    # Plotting options
    save_plots = FALSE,
    show_plots = TRUE,
    plot_subset = 0.1,
    plots_location = NA,
    
    # Dummy generation
    dummy_seed = NULL,
    
    # Matching options
    matching_p = 2,
    matching_fast = TRUE,
    
    # Remapping field plot options
    grid_density = 7,
    arrow_scale = 1,
    arrow_lwd = 0.8,
    arrow_length = .3,
    point_alpha = 1,
    time_steps_plot = 20,
    base_colors = NULL
) {
  
  
  verbose_training <- if(verbose_training == TRUE) 1 else 0
  
  # Output tables
  create_results_df <- function(n_classes, repetitions) {
    base_cols <- list(
      gauss_dist = numeric(repetitions),
      Accuracy_pw = numeric(repetitions),
      Computational_Time = numeric(repetitions),
      best_epoch = numeric(repetitions),
      Accuracy_iw = numeric(repetitions)
    )
    
    recall_cols <- setNames(
      as.list(rep(list(numeric(repetitions)), n_classes)),
      paste0("Recall_class_", 0:(n_classes-1))
    )
    
    as.data.frame(c(base_cols, recall_cols))
  }
  
  results <- create_results_df(repetitions = repetitions, n_classes = length(data_means))
  
  # Main loop
  for (i in 1:repetitions) {
    
    t0 <- Sys.time()
    
    seed <- i
    set_random_seed(seed)
    
    # Generate data for this repetition
    set.seed(dummy_seed)
    
    data <- generator(
      dimensions = dimensions,
      means = data_means,
      variances = data_variances,
      plot = show_plots)
    
    x_train <- data$x_train
    y_train <- data$y_train
    x_test <- data$x_test
    y_test <- data$y_test
    
    # Save Gaussian distance
    results$gauss_dist[i] <- mean(combn(seq_along(data_means), 2, function(j) wasserstein_gaussian(data_means[[j[1]]], data_variances[[j[1]]], data_means[[j[2]]], data_variances[[j[2]]]), simplify = TRUE))
    
    time <- system.time(fit <- Train_OT_model(
      x_train_real = x_train,
      y_train_real = y_train,
      x_test_real = x_test,
      y_test_real = y_test,
      mlp_units = mlp_units,
      mlp_dropout = mlp_dropout,
      epochs = epochs,
      patience = patience,
      verbose_training = verbose_training,
      matching_p = matching_p,
      matching_fast = matching_fast
    )
    )
    
    results$Computational_Time[i] <- time[3]
    
    results$Accuracy_pw[i] <- fit$accuracy
    results$best_epoch[i] <- fit$best_epoch
    

    # Recall
    sm_scores <- predict_softmax_OT_model(fit$model, x_new = x_test, dummy_means = fit$dummy_means, instance_wise = TRUE)
    
    pred_labels <- apply(sm_scores[[2]], 1, which.max) -1
    
    cm <- table(y_test, pred_labels)
    recalls <- diag(cm) / rowSums(cm)
    results$Accuracy_iw[i] <- sum(diag(cm))/sum(cm)
    results[i, paste0("Recall_class_", 0:(length(recalls)-1))] <- recalls
    
    results$best_epoch[i] <- fit$best_epoch
    
    if (save_plots) {
      ind <- sample_common_indices(y_train, subset = plot_subset)
      jpeg(
        filename = paste0(plots_location, "/new_plot_",
                          round(results$gauss_dist[i], 2), "_", i, ".jpg"),
        width = 800, height = 800
      )
      
      plot_remapping_field(
        model = fit$model,
        x_train_real = x_train,
        y_train_real = y_train,
        x_train_dummy = fit$x_train_dummy,
        y_train_dummy = fit$y_train_dummy,
        mean_train = apply(x_train, 3, mean),
        sd_train = apply(x_train, 3, sd),
        dummy_means = fit$dummy_means,
        grid_density = grid_density,
        time_steps = time_steps_plot,
        arrow_scale = arrow_scale,
        arrow_lwd = arrow_lwd,
        arrow_length = arrow_length,
        point_alpha = point_alpha,
        base_colors = base_colors,
        title = paste0("Remapping Field (Distance: ", round(results$gauss_dist[i], 2), ")")
      )
      
      
      dev.off()
    }
    cat("Repetition", i, "gaussian dist:", results$gauss_dist[i], "time:", difftime(Sys.time(), t0, units = "secs"),  "\n")
    print(paste("Completed repetition", i))
  }
  
  return(results)
}

# ... ---------------------------------------------------------------------

all_results <- list()

for (i in 1:2) {
  
  # x <- 6*i/((1+sqrt(2))*2)
  # means <- list(c(-x, 0), c(x, 0), c(0, x))
  # variances <- list(diag(2, 2), diag(2, 2), diag(2, 2))
  
  means <- list(c(-i, 0), c(i, 0))
  variances <- list(diag(2, 2), diag(2, 2))
  
  # Run the simulation study
  result <- OT_model_simulation_new(
    data_means = means,
    data_variances = variances,
    epochs = 50,
    repetitions = 2,
    save_plots = F,
    show_plots = T,
    dummy_seed = 123,
    matching_p = 2,
    matching_fast = TRUE,
    mlp_units = c(32, 32),
    mlp_dropout = 0.3,
    learning_rate = 1e-2,
    verbose_training = FALSE
  )
  
  
  # Store in a list
  all_results[[paste0("dist_", i)]] <- result
}

# STORE RESULTS -----------------------------------------------------------

params <- list(
  repetitions = 10,
  dimensions = c(90, 20, 2),
  mlp_units = c(32, 32),
  mlp_dropout = 0.3,
  epochs = 50,
  patience = 50,
  learning_rate = 1e-2
)

data_configurations <- list(
  dist_1 = list(
    means = list(c(-1, 0), c(1, 0)),
    variances = list(diag(2, 2), diag(2, 2))
  ),
  dist_2 = list(
    means = list(c(-2, 0), c(2, 0)),
    variances = list(diag(2, 2), diag(2, 2))
  ),
  dist_3 = list(
    means = list(c(-3, 0), c(3, 0)),
    variances = list(diag(2, 2), diag(2, 2))
  ),
  dist_4 = list(
    means = list(c(-4, 0), c(4, 0)),
    variances = list(diag(2, 2), diag(2, 2))
  )
)

# Save everything
saveRDS(
  list(
    results = all_results,
    parameters = params,
    data_configurations = data_configurations,
    session_info = sessionInfo()
  ),
  file = "RESULTS 2.rds"
)
