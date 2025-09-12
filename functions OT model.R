library(tensorflow)
library(keras3)
library(readr)
library(dplyr)
library(mvtnorm)
library(transport)


# Compute classwise matching ----------------------------------------------

# Computes an optimal classwise transport matching between dummy and real tensors
# using optimal transport for each class separately.
# This returns a reordered dummy tensor matched to the real data in feature space.
compute_classwise_matching <- function(dummy_tensor, real_tensor, labels_dummy, labels_real, p = 2, fast = TRUE) {
  method <- if (fast) "shortsimplex" else "primaldual"
  
  classes_dummy <- sort(unique(labels_dummy))
  classes_real  <- sort(unique(labels_real))
  
  if (!identical(as.numeric(classes_dummy), as.numeric(classes_real))) {
    warning("Mismatch in unique class labels between dummy and real tensors.")
  }
  
  common_classes <- intersect(classes_dummy, classes_real)
  dim_proj   <- dim(dummy_tensor)[3]
  time_steps <- dim(dummy_tensor)[2]
  
  # Flatten dummy and real tensors to [n_points_total, dim_proj]
  n_dummy_total <- dim(dummy_tensor)[1] * time_steps
  n_real_total  <- dim(real_tensor)[1] * time_steps
  
  dummy_tensor_reordered <- array(NA, dim = dim(dummy_tensor))  # output dummy tensor
  
  for (class in common_classes) {
    idx_dummy <- which(labels_dummy == class)
    idx_real  <- which(labels_real  == class)
    
    if (length(idx_dummy) == 0 || length(idx_real) == 0) next  # skip empty classes
    
    dummy_subset <- dummy_tensor[idx_dummy, , , drop = FALSE]
    real_subset  <- real_tensor[ idx_real, , , drop = FALSE]
    
    # Build flattened matrices: [n_instances * time_steps, dim_proj]
    dummy_mat <- matrix(NA, nrow = length(idx_dummy) * time_steps, ncol = dim_proj)
    real_mat  <- matrix(NA, nrow = length(idx_real)  * time_steps, ncol = dim_proj)
    
    for (d in 1:dim_proj) {
      dummy_mat[, d] <- as.vector(dummy_subset[, , d])
      real_mat[, d]  <- as.vector(real_subset[, , d])
    }
    
    # Convert to empirical distributions
    dummy_pp <- pp(dummy_mat)
    real_pp  <- pp(real_mat)
    
    # Compute transport plan
    tplan <- transport(a = dummy_pp, b = real_pp, p = p, method = method)
    
    # Convert linear indices to (i, t)
    dummy_rows <- tplan$from
    real_rows  <- tplan$to
    
    dummy_i <- ((dummy_rows - 1) %/% time_steps) + 1
    dummy_t <- ((dummy_rows - 1) %%  time_steps) + 1
    
    real_i <- ((real_rows - 1) %/% time_steps) + 1
    real_t <- ((real_rows - 1) %%  time_steps) + 1
    
    # Map class-specific indices back to global indices
    dummy_global_i <- idx_dummy[dummy_i]
    real_global_i  <- idx_real[real_i]
    
    # Move dummy points into the new positions defined by real (i, t)
    for (k in seq_along(dummy_rows)) {
      dummy_tensor_reordered[real_global_i[k], real_t[k], ] <- dummy_tensor[dummy_global_i[k], dummy_t[k], ]
    }
  }
  
  return(list(
    real_tensor  = real_tensor,               # unchanged
    dummy_tensor = dummy_tensor_reordered     # reordered point-by-point
  ))
}




# Train OT model ----------------------------------------------------------

# Trains a model to map real sequences to structured dummy point clouds.
# After standardization, it generates evenly spaced dummy clusters by class,
# aligns them to real data via optimal matching, and fits an MLP to learn this mapping.
# Returns the trained model, dummy data, and test accuracy.
train_OT_model <- function(
    x_train_real,
    y_train_real,
    x_test_real,
    y_test_real,
    mlp_units = c(32, 32),
    mlp_dropout = 0,
    epochs = 100,
    patience = epochs,
    OT_batch_size = length(y_train_real),
    dummy_means = NULL,
    dummy_variances = NULL,
    verbose_training = TRUE,
    matching_fast = TRUE,
    matching_p = 2,
    learning_rate = 1e-2)
{
  # normalise real data
  mean_train <- apply(x_train_real, 3, mean)
  sd_train   <- apply(x_train_real, 3, sd)
  sd_train[sd_train == 0] <- 1
  
  x_train_real_s <- sweep(sweep(x_train_real, 3, mean_train, "-"), 3, sd_train, "/")
  x_test_real_s  <- sweep(sweep(x_test_real, 3, mean_train, "-"), 3, sd_train, "/")
  
  y_train_ts <- matrix(y_train_real, nrow = dim(x_train_real)[1], ncol = dim(x_train_real)[2])
  y_test_ts <- matrix(y_test_real, nrow = dim(x_test_real)[1], ncol = dim(x_test_real)[2])
  
  
  n_train <- dim(x_train_real)[1]
  n_test  <- dim(x_test_real)[1]
  t       <- dim(x_train_real)[2]
  r       <- dim(x_train_real)[3]
  k       <- length(unique(y_train_real))
  fraq    <- 2 * pi / k
  radius  <- 3 * k
  
  # dummy means and variances

  if (is.null(dummy_means)){
    dummy_means <- lapply(0:(k - 1), function(i) {
    coords_2d <- c(radius * sin(i * fraq), radius * cos(i * fraq))
    if (r > 2) c(coords_2d, rep(0, r - 2)) else coords_2d[1:r]
    })}
  
  if (is.null(dummy_variances)){
  dummy_variances <- replicate(k, diag(1, r), simplify = FALSE)
  }
  
  # helper: ensure table reports all classes
  levels_train <- sort(unique(y_train_real))
  levels_test  <- sort(unique(y_test_real))
  
  combine_arrays <- function(lst, name) {
    lst <- Filter(Negate(is.null), lst)
    if (length(lst) == 0) stop("Nothing to combine for ", name)
    arrs <- lapply(lst, `[[`, name)
    if (length(arrs) == 1) return(arrs[[1]])
    do.call(abind::abind, c(arrs, list(along = 1)))
  }
  
  
#                     TRAIN MATCHING

  n_batches_train <- ceiling(length(y_train_real) / OT_batch_size)
  matched <- vector("list", n_batches_train)
  ind <- 1
  cat("Starting OT matching for training data...\n")
  start <- Sys.time()
  for (i in seq_len(n_batches_train)) {
    till <- min(OT_batch_size * i, length(y_train_real))
    cat(sprintf("  Train batch %d/%d (instances %d:%d)\n", i, n_batches_train, ind, till))
    
    instances_table <- table(factor(y_train_real[ind:till], levels = levels_train))
    
    dummy_train <- generator(
      dimensions = c(till - ind + 1, t, r),
      means = dummy_means,
      variances = dummy_variances,
      split_test = FALSE,
      instances_per_class = instances_table,
      plot = FALSE
    )
    
    match <- tryCatch({
      compute_classwise_matching(
        dummy_tensor = dummy_train$x_train,
        real_tensor  = x_train_real_s[ind:till, , ],
        labels_dummy = dummy_train$y_train,
        labels_real  = y_train_real[ind:till],
        fast = matching_fast,
        p = matching_p
      )
    }, error = function(e) {
      warning("compute_classwise_matching failed for train batch ", i, ": ", e$message)
      NULL
    })
    
    matched[[i]] <- match
    ind <- 1 + i * OT_batch_size
  }
  
  #                     TEST MATCHING
  
  n_batches_test <- ceiling(length(y_test_real) / OT_batch_size)
  matched_test <- vector("list", n_batches_test)
  ind <- 1
  cat("Starting OT matching for test data...\n")
  for (i in seq_len(n_batches_test)) {
    till <- min(OT_batch_size * i, length(y_test_real))
    cat(sprintf("  Test batch %d/%d (instances %d:%d)\n", i, n_batches_test, ind, till))
    
    instances_table <- table(factor(y_test_real[ind:till], levels = levels_test))
    
    dummy_test <- generator(
      dimensions = c(till - ind + 1, t, r),
      means = dummy_means,
      variances = dummy_variances,
      split_test = FALSE,
      instances_per_class = instances_table,
      plot = FALSE
    )
    
    match <- tryCatch({
      compute_classwise_matching(
        dummy_tensor = dummy_test$x_train,
        real_tensor  = x_test_real_s[ind:till, , ],
        labels_dummy = dummy_test$y_train,
        labels_real  = y_test_real[ind:till],
        fast = matching_fast,
        p = matching_p
      )
    }, error = function(e) {
      warning("compute_classwise_matching failed for test batch ", i, ": ", e$message)
      NULL
    })
    
    matched_test[[i]] <- match
    ind <- 1 + i * OT_batch_size
  }
  
#                     MLP
  
  cat("Building MLP model...\n")
  input_shape <- dim(x_train_real)[-1]
  build_matching_model <- function(input_shape, mlp_units, mlp_dropout = 0) {
    inputs <- layer_input(shape = input_shape)
    x <- inputs
    for (j in seq_along(mlp_units)) {
      x <- x %>%
        layer_dense(units = mlp_units[j], activation = "relu", name = paste0("MLP_layer_", j)) %>%
        layer_dropout(mlp_dropout)
    }
    outputs <- x %>% layer_dense(units = input_shape[2], activation = "linear", name = "linear_layer")
    keras_model(inputs, outputs)
  }
  
  matching_model <- build_matching_model(input_shape, mlp_units, mlp_dropout)
  matching_model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = learning_rate))
  
  early_stop <- callback_early_stopping(patience = patience, restore_best_weights = T)
  callbacks <- list(early_stop)
  
  cat("Combining batches and fitting MLP...\n")
  x_train_all <- combine_arrays(matched, "real_tensor")
  y_train_all <- combine_arrays(matched, "dummy_tensor")
  x_val_all   <- combine_arrays(matched_test, "real_tensor")
  y_val_all   <- combine_arrays(matched_test, "dummy_tensor")
  
  history <- matching_model %>%
    fit(
      x = x_train_all,
      y = y_train_all,
      batch_size = max(1, round(n_train / 10)),
      epochs = epochs,
      callbacks = callbacks,
      verbose = verbose_training,
      validation_data = list(x_val_all, y_val_all)
    )
  end <- Sys.time()
  elapsed <- end - start
  cat("Training finished.\n")
  best_epoch <- early_stop$best_epoch
  
  #                     PREDICT AND EVALUATE
  
  cat("Predicting on test set and evaluating accuracy...\n")
  preds <- predict(matching_model, x_test_real_s, verbose = verbose_training)
  
  AAA <- predict_softmax_matching(matching_model, x_test_real, dummy_means, T)
  predicted_classes <- apply(AAA[[1]], c(1,2), which.max) - 1
  acc <- mean(predicted_classes == y_test_ts)
  cat("Matching model accuracy:", acc, "\n")
  
  return(within(list(
    model = matching_model,
    accuracy = acc,
    history = history,
    dummy_means = dummy_means,
    x_train_dummy = x_train_all,
    y_train_dummy = y_train_all,
    x_test_dummy = x_val_all,
    y_test_dummy = y_val_all
  ), {
    time <- elapsed
    best_epoch <- best_epoch
  }))
}




# Predict softmax OT model ------------------------------------------------

# Given a trained matching model and dummy class centroids,
# this function predicts softmax-like class probabilities for new sequences
# by measuring distances to dummy class centroids in the mapped space.
predict_softmax_OT_model <- function(model, x_new, dummy_means, instance_wise = FALSE) {
  
  mean_train <- apply(x_new, 3, mean)
  sd_train <- apply(x_new, 3, sd)
  
  x_new_s <- sweep(x_new, 3, mean_train, "-")
  x_new_s <- sweep(x_new, 3, sd_train, "/")
  
  preds <- predict(model, x_new_s)
  
  # Allocate matrix for softmax scores
  n_instances <- dim(preds)[1]
  sequence_length <- dim(preds)[2]
  n_classes <- length(dummy_means)
  
  sm <- array(NA, dim = c(n_instances, sequence_length, n_classes))
  
  for (i in 1:n_instances) {
    for (j in 1:sequence_length) {
      for (f in 1:n_classes) {
        sm[i, j, f] <- exp(-sqrt(sum((preds[i, j, ] - dummy_means[[f]])^2)))
      }
    }
  }
  
  sums <- apply(sm, MARGIN = c(1, 2), FUN = sum)
  SUMS <- array(rep(sums, n_classes), dim = c(nrow(sums), ncol(sums), n_classes))
  
  softmax_scores <- sm / SUMS
  
  if (instance_wise) {
    sm_cw <- apply(sm, MARGIN = c(1, 3), FUN = prod)
    sums_cw <- apply(sm, MARGIN = 1, FUN = sum)
    softmax_scores_cw <- sm_cw / sums_cw
    
    return(list(
      softmax_scores = softmax_scores,
      softmax_scores_cw = softmax_scores_cw
    ))
  } else {
    return(softmax_scores)
  }
}



# Plot remapping field ----------------------------------------------------

# Visualizes the learned mapping from real to dummy space through a vector field.
# Arrows show how input points are remapped.
plot_remapping_field <- function(model,
                                 x_train_real,
                                 y_train_real,
                                 x_train_dummy,
                                 y_train_dummy,
                                 mean_train,
                                 sd_train,
                                 dummy_means,
                                 grid_density = 15,
                                 time_steps = 20,
                                 arrow_scale = 1.5,
                                 point_alpha = 0.6,
                                 arrow_lwd = 2,
                                 arrow_length = .3,
                                 base_colors = NULL,
                                 title = "Remapping Vector Field") {
  
  library(scales)         # For alpha()
  library(RColorBrewer)   # For color palettes
  
  # 1. Compute plot limits (include both real + dummy data)
  
  real_all  <- matrix(x_train_real, ncol = 2)
  dummy_all <- matrix(x_train_dummy, ncol = 2)
  
  xlim <- range(real_all[, 1], dummy_all[, 1])
  ylim <- range(real_all[, 2], dummy_all[, 2])
  
  # 2. Define grid for vector field (only over real data)

  real_only_xlim <- range(real_all[, 1])
  real_only_ylim <- range(real_all[, 2])
  
  x_seq <- seq(real_only_xlim[1], real_only_xlim[2], length.out = grid_density)
  y_seq <- seq(real_only_ylim[1], real_only_ylim[2], length.out = grid_density)
  grid_points <- expand.grid(x = x_seq, y = y_seq)
  n_points <- nrow(grid_points)
  
  # 3. Create input tensor for the model
 
  # [n_points, time_steps, 2]
  grid_tensor <- array(NA, dim = c(n_points, time_steps, 2))
  for (i in 1:n_points) {
    for (t in 1:time_steps) {
      grid_tensor[i, t, ] <- as.numeric(grid_points[i, ])
    }
  }
  
  # 4. Normalize with training stats

  grid_tensor_s <- sweep(grid_tensor, 3, mean_train, "-")
  grid_tensor_s <- sweep(grid_tensor_s, 3, sd_train, "/")
  

  # 5. Predict remapped values

  preds <- predict(model, grid_tensor_s)
  
  original_points <- matrix(NA, nrow = n_points, ncol = 2)
  mapped_points   <- matrix(NA, nrow = n_points, ncol = 2)
  for (i in 1:n_points) {
    original_points[i, ] <- colMeans(grid_tensor[i, , ])
    mapped_points[i, ]   <- colMeans(preds[i, , ])
  }
  

  # 6. Arrow colors (per predicted position)

  n_classes <- length(dummy_means)
  
  if (is.null(base_colors)) {
    base_colors <- setNames(brewer.pal(n_classes, "Set1")[1:n_classes], 0:(n_classes - 1))
  }
  
  arrow_colors <- character(n_points)
  for (i in 1:n_points) {
      dists <- sapply(dummy_means, function(mu) sqrt(sum((mapped_points[i, ] - mu)^2)))
      closest_class <- which.min(dists) - 1
      arrow_colors[i] <- base_colors[as.character(closest_class)]
  }
  

  # 7. Plot

  plot(NULL, xlim = xlim, ylim = ylim, xlab = "X", ylab = "Y", asp = 1, main = title)
  
  # Color points by class label
  real_colors  <- base_colors[as.character(y_train_real)]
  dummy_colors <- base_colors[as.character(y_train_dummy)]
  
  # Plot real and dummy points (t = 1 only), using solid dots (pch = 16)
  points(matrix(x_train_real[, 1, ], ncol = 2),
         col = alpha(real_colors, point_alpha), pch = 16, cex = 1)
  points(matrix(x_train_dummy[, 1, ], ncol = 2),
         col = alpha(dummy_colors, point_alpha), pch = 16, cex = 1)
  
  # Arrows
  arrows(
    x0 = original_points[, 1],
    y0 = original_points[, 2],
    x1 = original_points[, 1] + arrow_scale * (mapped_points[, 1] - original_points[, 1]),
    y1 = original_points[, 2] + arrow_scale * (mapped_points[, 2] - original_points[, 2]),
    col = arrow_colors,
    lwd = arrow_lwd,
    length = arrow_length
  )
  
  # Legend
  legend("topright", legend = names(base_colors), col = base_colors, pch = 16, title = "Class Color")
}



