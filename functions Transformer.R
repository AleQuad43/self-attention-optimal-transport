library(tensorflow)
library(keras3)
library(readr)
library(dplyr)
library(mvtnorm)
library(transport)


# Generator ---------------------------------------------------------------

# creates synthetic data for multiple classes
generator <- function(dimensions = c(900, 10, 2), means, variances, 
                      instances_per_class = NULL, plot = TRUE, split_test = TRUE) {
  
  k <- length(means)
  
  # Determine instances per class
  if (is.null(instances_per_class)) {
    if (dimensions[1] %% k != 0) stop("Instances must be divisible by number of classes")
    instances_per_class <- rep(dimensions[1] / k, k)
  } else if (length(instances_per_class) == 1) {
    instances_per_class <- rep(instances_per_class, k)
  } else if (length(instances_per_class) != k) {
    stop("Length of instances_per_class must match number of classes (k) or 1.")
  }
  
  total_instances <- dimensions[1]
  seq_len <- dimensions[2]
  feat_dim <- dimensions[3]
  
  if (sum(instances_per_class) != total_instances) {
    stop("SUm of instances_per_class must match number of instances (first argument in dimensions).")
  }
  # Labels
  y <- rep(0:(k-1), times = instances_per_class)
  
  # Allocate array
  x <- array(NA, c(total_instances, seq_len, feat_dim))
  
  # Generate sequences class by class
  counter <- 1
  for (class_idx in 1:k) {
    n_inst <- instances_per_class[class_idx]
    if (n_inst == 0) next 
    # Generate one class's sequences at once
    samples <- array(NA, c(n_inst, seq_len, feat_dim))
    for (t in 1:seq_len) {
      samples[, t, ] <- rmvnorm(n_inst, mean = means[[class_idx]], sigma = variances[[class_idx]])
    }
    
    x[counter:(counter+n_inst-1), , ] <- samples
    counter <- counter + n_inst
  }
  
  # Train/test split
  if (split_test) {
    index <- seq(1, total_instances, by = 2)
    x_train <- x[index, , ]
    y_train <- y[index]
    x_test  <- x[-index, , ]
    y_test  <- y[-index]
  } else {
    x_train <- x; y_train <- y
    x_test <- NULL; y_test <- NULL
  }
  
  if (plot) scatter(x_train, y_train, title = "Training set scatter plot")
  
  return(list(x_train = x_train, y_train = y_train,
              x_test  = x_test,  y_test = y_test))
}


# Plotting functions ------------------------------------------------------

# Plots 2D scatter of the data colored by class
scatter <- function(data, labels, title = "Scatter Plot of train set", 
                    set_plot_xlim = NULL, set_plot_ylim = NULL, colors = NULL) {
  instances <- dim(data)[1]
  time_steps <- dim(data)[2]
  unique_labels <- unique(labels)  # Get unique label values
  
  # Assign default colors if none are provided
  if (is.null(colors)) {
    color_palette <- rainbow(length(unique_labels))  # Generate distinct colors
    names(color_palette) <- unique_labels  # Assign each label a color
  } else {
    color_palette <- colors
  }
  
  # Automatically determine xlim and ylim if not provided
  if (is.null(set_plot_xlim)) {
    set_plot_xlim <- range(data[,,1], na.rm = TRUE)
  }
  if (is.null(set_plot_ylim)) {
    set_plot_ylim <- range(data[,,2], na.rm = TRUE)
  }
  
  # Initialize empty plot
  plot(NULL, xlim = set_plot_xlim, ylim = set_plot_ylim, 
       xlab = paste("Feature", 1), ylab = paste("Feature", 2), main = title, pch = 16)
  
  # Plot each instance over time steps
  for (i in 1:instances) {
    color <- color_palette[as.character(labels[i])]  # Assign color dynamically
    points(data[i, , 1], data[i, , 2], col = color, pch = 16) 
  }
  
  # Add legend
  legend("topright", legend = unique_labels, col = color_palette, pch = 16, title = "Labels")
}

# Animates projections across transformer blocks, optionally with MLP predictions
dynamic_plot_wg <- function(projections, y_train, delay = 1, MLP_predictions = NULL) {
  n_proj <- sum(sapply(projections, Negate(is.null)))
  time_steps <- dim(projections[[1]])[2]
  
  unique_labels <- sort(unique(y_train))
  colors <- setNames(rainbow(length(unique_labels)), unique_labels)
  mlp_colors <- setNames(rainbow(length(unique_labels), alpha = 0.2), unique_labels)  # transparent background
  
  dev.new()
  plot.new()
  
  for (i in 1:n_proj) {
    proj <- projections[[i]]
    
    # ---- Background MLP prediction surface ----
    if (!is.null(MLP_predictions) && !is.null(MLP_predictions[[i]])) {
      df_pred <- MLP_predictions[[i]]
      
      x_vals <- sort(unique(df_pred[[1]]))
      y_vals <- sort(unique(df_pred[[2]]))
      n_x <- length(x_vals)
      n_y <- length(y_vals)
      
      # Convert predictions to matrix for raster plotting
      pred_matrix <- matrix(df_pred$prediction, nrow = n_x, ncol = n_y)
      
      image(
        x = x_vals,
        y = y_vals,
        z = pred_matrix,
        col = mlp_colors,
        useRaster = TRUE,
        xlab = paste("Feature", 1),
        ylab = paste("Feature", 2),
        main = paste("Epoch:", i, "- MLP + Residual Projections"),
        xlim = range(proj[,,1], na.rm = TRUE),
        ylim = range(proj[,,2], na.rm = TRUE)
      )
    } else {
      plot(NULL,
           xlim = range(proj[,,1], na.rm = TRUE), 
           ylim = range(proj[,,2], na.rm = TRUE), 
           xlab = paste("Feature", 1), 
           ylab = paste("Feature", 2), 
           main = paste("Epoch:", i, "- Residual Projections"))
    }
    
    # ---- Overlay projection points ----
    for (label in unique_labels) {
      label_index <- which(y_train == label)
      
      for (t in 1:time_steps) {
        points(proj[label_index, t, 1],
               proj[label_index, t, 2],
               col = colors[as.character(label)],
               pch = 16)
      }
    }
    
    Sys.sleep(delay)
  }
}

# shows mapping from training data (start) to final projection (end of training). Uses arrows to match each point
plot_transformer_mapping <- function(proj, labels, ind, xlim = NULL, ylim = NULL, name = "Transformer mapping (start → end)") {
  
  classes <- sort(unique(labels))
  colors <- rainbow(length(classes))
  names(colors) <- classes
  
  start_proj <- proj[[1]][ind,,]
  end_proj <- proj[[length(proj)]][ind,,]
  
  lab <- labels[ind]
  
  if (is.null(xlim)) {xlim <- range(start_proj[,,1], end_proj[,,1])}
  if (is.null(ylim)) {ylim <- range(start_proj[,,2], end_proj[,,2])}
  
  plot(NULL, xlim = xlim,  ylim = ylim,
       xlab = "X", ylab = "Y", main = name)
  
  for (class in classes) {
    indices <- which(lab == class)
    
    S <- start_proj[indices, , ,drop = FALSE]
    D <- end_proj[indices, , ,drop = FALSE]
    
    S_mat <- matrix(NA, nrow = length(indices) * dim(S)[2], ncol = 2)
    D_mat <- matrix(NA, nrow = length(indices) * dim(D)[2], ncol = 2)
    
    for (d in 1:2) {
      S_mat[, d] <- as.vector(S[, , d])
      D_mat[, d] <- as.vector(D[, , d])
    }
    
    col <- colors[as.character(class)]
    
    points(S_mat, col = col, pch = 1)
    points(D_mat, col = col, pch = 16)
    
    arrows(S_mat[, 1], S_mat[, 2], D_mat[, 1], D_mat[, 2], 
           col = adjustcolor(col, alpha.f = 0.2), length = 0.05)
  }
  
  legend("topright", legend = classes, col = colors, pch = 16, title = "Classes")
}

# shows the full trajectory of each point across all epochs (not just start → end).
plot_transformer_paths <- function(proj, labels, ind, xlim = NULL, ylim = NULL, name = "Transformer paths (start → end)") {
  
  classes <- sort(unique(labels))
  
  n_steps <- length(proj)
  n_samples <- dim(proj[[1]][ind,,])[1]
  time_steps <- dim(proj[[1]])[2]
  n_features <- dim(proj[[1]])[3]
  
  lab <- labels[ind]
  
  classes <- sort(unique(lab))
  colors <- rainbow(length(classes))
  names(colors) <- classes
  
  if (is.null(xlim)) {xlim <- range(proj[[1]][ind, , 1], proj[[n_steps]][ind, , 1])}
  if (is.null(ylim)) {ylim <- range(proj[[1]][ind, , 2], proj[[n_steps]][ind, , 2])}
  
  plot(NULL, xlim = xlim, ylim = ylim, xlab = "X", ylab = "Y", main = name)
  
  # background
  rest <- setdiff(seq_along(labels), ind)
  l_rest <- labels[rest]
  
  if (length(rest) != 0) {
    for (i in seq_along(rest)) {
      idx <- rest[i]
      class <- labels[idx]
      col <- adjustcolor(colors[as.character(class)], alpha.f = 0.3)
      
      # plot start and end points (mean over time_steps)
      x_start <- mean(proj[[1]][idx, , 1])
      y_start <- mean(proj[[1]][idx, , 2])
      
      x_end <- mean(proj[[n_steps]][idx, , 1])
      y_end <- mean(proj[[n_steps]][idx, , 2])
      
      points(x_start, y_start, col = col, pch = 1, cex = 1)
      points(x_end, y_end, col = col, pch = 16, cex = 1)
    }
  }
  
  for (i in 1:n_samples) {
    class <- lab[i]
    col <- colors[as.character(class)]
    true_index <- ind[i]
    
    for (j in 1:time_steps) {
      path <- matrix(NA, nrow = n_steps, ncol = n_features)
      for (t in 1:n_steps) {
        path[t, ] <- proj[[t]][true_index, j, ]
      }
      lines(path, col = adjustcolor(col, alpha.f = 0.3))
      points(path[1, 1], path[1, 2], col = col, pch = 1, cex = 1)
      points(path[n_steps, 1], path[n_steps, 2], col = col, pch = 16, cex = 1)
    }
  }
  
  legend("topright", legend = classes, col = colors, pch = 16, title = "Classes")
}

# samples a fraction of indices per class, useful for making plots clearer and ensuring consistency across plots
sample_common_indices <- function(labels, subset = 1, seed = NA) {
  
  if (!is.numeric(subset) || length(subset) != 1 || subset < 0 || subset > 1) {
    stop("`subset` must be a numeric value between 0 and 1")
  }
  
  classes <- sort(unique(labels))
  ind <- c()
  for (j in seq_along(classes)) {
    points <- which(labels == classes[j])
    a <- sample(points, round(length(points) * subset))
    ind <- c(ind, a)
  }
  return(ind)
}

# adds a table of computed metrics (Wasserstein distance, Transformer distance, Transformer path cost) to an existing plot.
add_cost_table <- function(proj, labels, corner = "bottomright", cex = 0.8, inset = 0.02,
                           box_col = NA,
                           wasserstein_info = NULL,
                           transformer_info = NULL,
                           stepwise_cost = NULL) {
  
  # Use precomputed or compute fresh
  if (is.null(transformer_info)) {
    transformer_info <- compute_transformer_distance(proj, labels)
  }
  if (is.null(wasserstein_info)) {
    wasserstein_info <- compute_wasserstein_distance(proj, labels)
  }
  if (is.null(stepwise_cost)) {
    stepwise_cost <- compute_transformer_cost(proj)
  }
  
  transformer_cost <- transformer_info$total_cost
  wass_total <- wasserstein_info$total_wass
  
  cost_table <- data.frame(
    Metric = c("Wasserstein", "Transformer (direct)", "Transformer (stepwise)"),
    Cost = sprintf("%.2f", c(wass_total, transformer_cost, stepwise_cost))
  )
  
  labels_text <- paste(format(cost_table$Metric, width = 24), cost_table$Cost)
  
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  
  par(xpd = TRUE)
  
  legend(corner,
         legend = labels_text,
         bty = ifelse(is.na(box_col), "n", "o"),
         cex = cex,
         inset = inset,
         x.intersp = 0.6,
         y.intersp = 1.2,
         text.col = "black",
         bg = box_col
  )
}



# Compute metrics ---------------------------------------------------------

# computes class-wise Euclidean distance between training set and final projection (direct mapping).
compute_transformer_distance <- function(proj, labels) {
  start_proj <- proj[[1]]
  end_proj <- proj[[length(proj)]]
  
  unique_classes <- sort(unique(labels))
  dim_proj <- dim(start_proj)[3]  # Number of spatial dimensions
  cost_per_class <- numeric(length(unique_classes))
  names(cost_per_class) <- unique_classes
  
  for (i in seq_along(unique_classes)) {
    class <- unique_classes[i]
    indices <- which(labels == class)
    
    S <- start_proj[indices, , ]
    D <- end_proj[indices, , ]
    
    S_mat <- matrix(NA, nrow = length(indices) * dim(S)[2], ncol = dim_proj)
    D_mat <- matrix(NA, nrow = length(indices) * dim(D)[2], ncol = dim_proj)
    
    for (d in 1:dim_proj) {
      S_mat[, d] <- as.vector(S[, , d])
      D_mat[, d] <- as.vector(D[, , d])
    }
    
    proj_S <- pp(S_mat)
    proj_D <- pp(D_mat)
    
    cost_per_class[i] <- sqrt(sum((proj_S$coordinates - proj_D$coordinates)^2))
  }
  
  total_cost <- sum(cost_per_class)
  return(list(per_class_cost = cost_per_class, total_cost = total_cost))
}

# computes Wasserstein distance between start and end projections by solving the optimal transport problem.
compute_wasserstein_distance <- function(proj, labels, p = 2, fast = TRUE) {
  start_proj <- proj[[1]]
  end_proj <- proj[[length(proj)]]
  method <- if (fast) "shortsimplex" else "primaldual"
  
  unique_classes <- sort(unique(labels))
  dim_proj <- dim(start_proj)[3]
  
  wass_per_class <- numeric(length(unique_classes))
  names(wass_per_class) <- unique_classes
  
  transport_plans <- list()  # Store transport plans
  
  for (i in seq_along(unique_classes)) {
    class <- unique_classes[i]
    indices <- which(labels == class)
    
    S <- start_proj[indices, , ]
    D <- end_proj[indices, , ]
    
    S_mat <- matrix(NA, nrow = length(indices) * dim(S)[2], ncol = dim_proj)
    D_mat <- matrix(NA, nrow = length(indices) * dim(D)[2], ncol = dim_proj)
    
    for (d in 1:dim_proj) {
      S_mat[, d] <- as.vector(S[, , d])
      D_mat[, d] <- as.vector(D[, , d])
    }
    
    proj_S <- pp(S_mat)
    proj_D <- pp(D_mat)
    
    # Compute optimal transport plan
    tplan <- suppressMessages(
      invisible(
        transport(a = proj_S, b = proj_D, p = p, method = method)
      )
    )
    
    transport_plans[[as.character(class)]] <- tplan
    
    # Compute Wasserstein distance
    wass <- wasserstein(a = proj_S, b = proj_D, tplan = tplan, p = p, prob = FALSE)
    wass_per_class[i] <- wass
  }
  
  total_wass <- sum(wass_per_class)
  
  return(list(
    per_class_wass = wass_per_class,
    total_wass = total_wass,
    transport_plans = transport_plans  # Class-indexed list of transport plans
  ))
}

# computes cumulative path length covered by each point across all training steps.
compute_transformer_cost <- function(proj, single_points = FALSE) {
  n_steps <- length(proj)
  n_samples <- dim(proj[[1]])[1]
  time_steps <- dim(proj[[1]])[2]
  n_features <- dim(proj[[1]])[3]
  
  indices <- 1:n_samples
  
  distance <- numeric(length(indices))  # One distance per point
  
  for (k in seq_along(indices)) {
    i <- indices[k]
    total_distance <- 0
    
    for (j in 1:time_steps) {
      path <- matrix(NA, nrow = n_steps, ncol = n_features)
      
      for (t in 1:n_steps) {
        path[t, ] <- proj[[t]][i, j, ]
      }
      
      diffs <- diff(path)  # Differences between consecutive steps
      dists <- sqrt(rowSums(diffs^2))
      total_distance <- total_distance + sum(dists)
    }
    
    distance[k] <- total_distance
  }
  
  if (single_points == FALSE) {
    distance <- sum(distance)
  }
  
  return(distance)
}

# closed-form Wasserstein distance between two multivariate Gaussians
wasserstein_gaussian <- function(mu1, Sigma1, mu2, Sigma2) {
  diff <- mu1 - mu2
  term1 <- sum(diff^2)
  
  sqrt_Sigma2 <- sqrtm(Sigma2)
  inner_term <- sqrtm(sqrt_Sigma2 %*% Sigma1 %*% sqrt_Sigma2)
  
  term2 <- sum(diag(Sigma1 + Sigma2 - 2 * inner_term))
  
  return(sqrt(term1 + term2))
}



# Models ------------------------------------------------------------------

# builds a transformer-based model with multi-head attention, feedforward blocks, residual connections, and MLP classifier head.
build_model <- function(input_shape,
                        head_size,
                        num_heads,
                        ff_dim,
                        num_transformer_blocks,
                        mlp_units,
                        dropout = 0,
                        mlp_dropout = 0,
                        y_train) {
  
  transformer_encoder <- function(inputs,
                                  head_size,
                                  num_heads,
                                  ff_dim,
                                  dropout = 0) {
    
    attention_layer <-
      layer_multi_head_attention(key_dim = head_size,
                                 num_heads = num_heads,
                                 dropout = dropout)
    
    n_features <- inputs$shape[[3]]
    
    x <- inputs %>%
      attention_layer(., .) %>%
      layer_dropout(dropout) %>%
      layer_layer_normalization(epsilon = 1e-6)
    
    res <- x + inputs
    
    x <- res %>%
      layer_conv_1d(ff_dim, kernel_size = 1, activation = "relu") %>%
      layer_dropout(dropout) %>%
      layer_conv_1d(n_features, kernel_size = 1) %>%
      layer_layer_normalization(epsilon = 1e-6)
    
    list(output = x + res, res = res, res_after_FF = x + res)
  }
  
  inputs <- layer_input(input_shape)
  
  x <- inputs
  res_layers <- list()
  res_aFF_layers <- list()
  
  for (i in 1:num_transformer_blocks) {
    block_output <- transformer_encoder(
      x,
      head_size = head_size,
      num_heads = num_heads,
      ff_dim = ff_dim,
      dropout = dropout
    )
    
    x <- block_output$output
    res_layers[[i]] <- block_output$res
    res_aFF_layers[[i]] <- block_output$res_after_FF
  }
  
  for (i in seq_along(mlp_units)) {
    x <- x %>%
      layer_dense(units = mlp_units[i], activation = "relu", name = paste0("MLP_layer_", i)) %>%
      layer_dropout(mlp_dropout)
  }
  
  outputs <- x %>% 
    layer_dense(length(unique(y_train)), activation = "softmax", name = "MLP_layer_sm")
  
  keras_model(inputs, c(outputs, res_layers, res_aFF_layers))
}

# builds a simple MLP classifier. Used for pre-training before transformer training if enabled.
build_mlp_model <- function(input_shape,
                            mlp_units,
                            mlp_dropout = 0,
                            y_train) {
  inputs <- layer_input(shape = input_shape)
  x <- inputs
  
  for (i in seq_along(mlp_units)) {
    x <- x %>%
      layer_dense(units = mlp_units[i], activation = "relu", name = paste0("MLP_layer_", i)) %>%
      layer_dropout(rate = mlp_dropout)
  }
  
  outputs <- x %>%
    layer_dense(units = length(unique(y_train)), activation = "softmax", name = "MLP_output")
  
  keras_model(inputs = inputs, outputs = outputs)
}

# FitTransformer: main training function.
# - Standardizes data
# - Optionally pre-trains MLP (if pretrain = TRUE)
# - Builds and trains transformer model
# - Optionally saves intermediate projections for visualization
fit_transformer <- function(
    x_train,
    y_train,
    x_test,
    y_test,
    transformer_blocks = 1,
    transformer_head_size = 16,
    transformer_num_heads = 10,
    transformer_ff_dim = 32,
    transformer_dropout = 0.1,
    mlp_units = c(8),
    mlp_dropout = 0.4,
    epochs = 100,
    patience = epochs,
    learning_rate = 1e-2,
    verbose_training = TRUE,
    pretrain = FALSE,
    pretrain_means = NULL,
    pretrain_variances = NULL,
    save_projections = FALSE
) {
  
  mean_train <- apply(x_train, 3, mean)
  sd_train   <- apply(x_train, 3, sd)
  
  x_train_s <- sweep(sweep(x_train, 3, mean_train, "-"), 3, sd_train, "/")
  x_test_s  <- sweep(sweep(x_test, 3, mean_train, "-"), 3, sd_train, "/")
  
  y_train_ts <- matrix(y_train, nrow = dim(x_train)[1], ncol = dim(x_train)[2])
  y_test_ts  <- matrix(y_test,  nrow = dim(x_test)[1],  ncol = dim(x_test)[2])
  
  if (pretrain == T){
    r       <- dim(x_train)[3]
    k       <- length(unique(y_train))
    fraq    <- 2 * pi / k
    radius  <- 3 * k
    if (is.null(pretrain_means)){
      pretrain_means <- lapply(0:(k - 1), function(i) {
        coords_2d <- c(radius * sin(i * fraq), radius * cos(i * fraq))
        if (r > 2) c(coords_2d, rep(0, r - 2)) else coords_2d[1:r]
      })}
    
    if (is.null(pretrain_variances)){
      pretrain_variances <- replicate(k, diag(1, r), simplify = FALSE)
    }
    
    sim <- generator(dimensions = dim(x_train),
                     means = pretrain_means, variances = pretrain_variances, plot = F)
    
    sim$mean_train <- apply(sim$x_train, 3, mean)
    sim$sd_train <- apply(sim$x_train, 3, sd)
    sim$x_train_s <- sweep(sweep(sim$x_train, 3, sim$mean_train, "-"), 3, sim$sd_train, "/")
    sim$x_test_s <- sweep(sweep(sim$x_test, 3, sim$mean_train, "-"), 3, sim$sd_train, "/")
    sim$y_train_ts <- matrix(sim$y_train, nrow = dim(sim$x_train)[1], ncol = dim(sim$x_train)[2])
    sim$y_test_ts <- matrix(sim$y_test, nrow = dim(sim$x_test)[1], ncol = dim(sim$x_test)[2])
    
    mlp_model <- build_mlp_model(input_shape = dim(sim$x_train)[-1], mlp_units = mlp_units,
                                 mlp_dropout = mlp_dropout, y_train = sim$y_train)
    
    mlp_model %>% compile(
      loss = "sparse_categorical_crossentropy",
      optimizer = optimizer_adam(learning_rate = learning_rate),
      metrics = "accuracy"
    )
    
    callbacks <- list(callback_early_stopping(patience = 5, restore_best_weights = TRUE))
    
    mlp_model %>%
      fit(x = sim$x_train, y = sim$y_train_ts,
          batch_size = dim(sim$x_test)[1],
          shuffle = FALSE, epochs = epochs,
          callbacks = callbacks,
          validation_data = list(sim$x_test_s, sim$y_test_ts),
          verbose = F)
    
    save_model_weights(mlp_model, filepath = "MLP.weights.h5", overwrite = TRUE)
  }
  
  model <- build_model(
    input_shape = dim(x_train)[-1],
    head_size = transformer_head_size,
    num_heads = transformer_num_heads,
    ff_dim = transformer_ff_dim,
    num_transformer_blocks = transformer_blocks,
    mlp_units = mlp_units,
    mlp_dropout = mlp_dropout,
    dropout = transformer_dropout,
    y_train = y_train
  )
  
  model %>% compile(
    loss = c("sparse_categorical_crossentropy", rep(NULL, 2 * transformer_blocks)),
    optimizer = optimizer_rmsprop(learning_rate = learning_rate),
    metrics = c(list("accuracy"), rep(list(NULL), 2 * transformer_blocks))
  )
  
  if (pretrain) {
    load_model_weights(model, filepath = "MLP.weights.h5", skip_mismatch = TRUE)
    freeze_weights(model, from = "MLP_layer_1")
  }
  
  if (save_projections) {
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
  } else {
    early_stop <- callback_early_stopping(patience = patience, restore_best_weights = TRUE)
    callbacks <- list(early_stop)
    
  }
  
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
  
  return(within(list(
    model   = model,
    history = history
  ), {
    # always include these
    time       <- training_time[["elapsed"]]   # cleaner than [3]
    best_epoch <- early_stop$best_epoch
    
    # optionally include projections if requested
    if (save_projections) {
      proj <- projections[[transformer_blocks]][1:best_epoch]
    } else {
      proj <- NULL
    }
  }))
  
}
