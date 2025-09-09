rm(list=ls()); gc(); graphics.off(); cat("\014")

library(dplyr)
library(xgboost)
library(Metrics)
library(terra)
library(ggplot2)
library(SHAPforxgboost)
library(gridExtra)
library(patchwork)

set.seed(2025)

data_csv      <- "data/Model_Data_RSEI_means.csv"
stack_path    <- "data/PredictorStack.tif"
model_dir     <- "models"
out_dir       <- "outputs"
fig_dir       <- "figures"

dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(out_dir,   showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir,   showWarnings = FALSE, recursive = TRUE)

df <- read.csv(data_csv)
if (all(c("AgriInt","PopDens") %in% names(df))) df <- df %>% select(-AgriInt, -PopDens)

df <- df %>% mutate(log_Dist_BU = log1p(Dist_BU), log_Dist_Road = log1p(Dist_Road))

cont_vars <- c("Elev","Slope","LST_mean","PPT_mean","SM_mean","VHI_mean")
scale_stats <- df %>% summarise(across(all_of(cont_vars), list(mean = ~mean(.), sd = ~sd(.))))
scale_means <- scale_stats %>% select(ends_with("_mean")) %>% as.numeric(); names(scale_means) <- cont_vars
scale_sds   <- scale_stats %>% select(ends_with("_sd"))   %>% as.numeric(); names(scale_sds)   <- cont_vars

df <- df %>% mutate(across(all_of(cont_vars), ~(. - scale_means[cur_column()]) / scale_sds[cur_column()] ))

X <- df %>% select(log_Dist_BU, log_Dist_Road, all_of(cont_vars))
y <- df$RSEI_trend

dtrain <- xgb.DMatrix(data = as.matrix(X), label = y)
n_cores <- max(1, parallel::detectCores() - 1)

params <- list(
  objective        = "reg:squarederror",
  eta              = 0.05,
  max_depth        = 8,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  lambda           = 1,
  alpha            = 0.5,
  nthread          = n_cores
)

cv <- xgb.cv(
  params                = params,
  data                  = dtrain,
  nrounds               = 1000,
  nfold                 = 5,
  early_stopping_rounds = 50,
  metrics               = "rmse",
  verbose               = 0
)

best_nrounds <- cv$best_iteration

xgb_model <- xgb.train(
  params       = params,
  data         = dtrain,
  nrounds      = best_nrounds,
  watchlist    = list(train = dtrain),
  print_every_n= 0
)

y_pred <- predict(xgb_model, dtrain)
metrics <- list(
  RMSE = rmse(y, y_pred),
  MAE  = mae(y, y_pred),
  R2   = 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2),
  MAPE = mean(abs((y - y_pred)/y))*100,
  Bias = mean(y - y_pred)
)
print(metrics)

xgb.save(xgb_model, file.path(model_dir, "xgb_rsei.model"))
saveRDS(list(means = scale_means, sds = scale_sds), file.path(model_dir, "scale_params_rsei.rds"))

full_stack <- rast(stack_path)
names(full_stack) <- c(
  "RSEI_trend","AgriInt","Dist_BU","Dist_Road",
  "Elev","Slope","PopDens","LST_mean","PPT_mean","SM_mean","VHI_mean"
)

r_template <- full_stack[["RSEI_trend"]]
pred_stack <- full_stack[[c("Dist_BU","Dist_Road", cont_vars)]]
pred_stack$log_Dist_BU   <- log1p(pred_stack$Dist_BU)
pred_stack$log_Dist_Road <- log1p(pred_stack$Dist_Road)
pred_stack <- pred_stack[[c("log_Dist_BU","log_Dist_Road", cont_vars)]]

scale_list  <- readRDS(file.path(model_dir, "scale_params_rsei.rds"))
s_means     <- scale_list$means
s_sds       <- scale_list$sds
model       <- xgb.load(file.path(model_dir, "xgb_rsei.model"))

pred_fun <- function(logDistBU, logDistRoad, Elev, Slope, LST_mean, PPT_mean, SM_mean, VHI_mean) {
  df_block <- data.frame(
    logDistBU   = logDistBU,
    logDistRoad = logDistRoad,
    Elev        = (Elev     - s_means["Elev"])     / s_sds["Elev"],
    Slope       = (Slope    - s_means["Slope"])    / s_sds["Slope"],
    LST_mean    = (LST_mean - s_means["LST_mean"]) / s_sds["LST_mean"],
    PPT_mean    = (PPT_mean - s_means["PPT_mean"]) / s_sds["PPT_mean"],
    SM_mean     = (SM_mean  - s_means["SM_mean"])  / s_sds["SM_mean"],
    VHI_mean    = (VHI_mean - s_means["VHI_mean"]) / s_sds["VHI_mean"]
  )
  as.numeric(predict(model, xgb.DMatrix(as.matrix(df_block), missing = NA)))
}

r_pred <- terra::lapp(
  x        = pred_stack,
  fun      = pred_fun,
  filename = file.path(out_dir, "xgb_rsei_trend_pred.tif"),
  overwrite= TRUE
)

r_pred_masked <- mask(r_pred, r_template)
writeRaster(r_pred_masked, file.path(out_dir, "xgb_rsei_trend_pred_masked.tif"), overwrite = TRUE)

m <- matrix(c(-Inf,-5,1,-5,0,2,0,5,3,5,Inf,4), ncol = 3, byrow = TRUE)
r_classified <- classify(r_pred_masked, rcl = m, include.lowest = TRUE, right = FALSE)
writeRaster(r_classified, file.path(out_dir, "xgb_rsei_trend_classes.tif"), overwrite = TRUE)

df_eval <- data.frame(Observed = y, Predicted = y_pred)
p_obs <- ggplot(df_eval, aes(Observed, Predicted)) +
  geom_point(alpha = 0.4) + geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  theme_minimal() + coord_equal()
ggsave(file.path(fig_dir, "observed_vs_predicted.png"), p_obs, dpi = 500, width = 8, height = 6, bg = "white")

set.seed(2025)
sample_idx <- sample(1:nrow(X), size = min(3000, nrow(X)))
X_sample   <- X[sample_idx, ]
y_sample   <- y[sample_idx]

shap_result <- shap.values(xgb_model = xgb_model, X_train = as.matrix(X_sample))
shap_long   <- shap.prep(shap_contrib = shap_result$shap_score, X_train = as.matrix(X_sample))

png(file.path(fig_dir, "shap_bar.png"), width = 1800, height = 1200, res = 300)
shap.plot.summary(shap_long, kind = "bar")
dev.off()

png(file.path(fig_dir, "shap_sina.png"), width = 1800, height = 1200, res = 300)
shap.plot.summary(shap_long, kind = "sina")
dev.off()

shap_importance <- sort(shap_result$mean_shap_score, decreasing = TRUE)
top_features <- names(shap_importance)[1:min(6, length(shap_importance))]
fig_list <- lapply(top_features, shap.plot.dependence, data_long = shap_long, dilute = 5)
png(file.path(fig_dir, "shap_dependence_grid.png"), width = 2000, height = 1600, res = 300)
gridExtra::grid.arrange(grobs = fig_list, ncol = 2)
dev.off()

B <- 20
pred_matrix <- matrix(NA, nrow = nrow(X_sample), ncol = B)
for (b in 1:B) {
  boot_idx <- sample(1:nrow(X_sample), replace = TRUE)
  X_boot   <- as.matrix(X_sample[boot_idx, ])
  y_boot   <- y_sample[boot_idx]
  dtrain_b <- xgb.DMatrix(data = X_boot, label = y_boot)
  model_b  <- xgb.train(
    params = list(objective="reg:squarederror", eta=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, lambda=1, alpha=0.5, nthread=1),
    data   = dtrain_b,
    nrounds= 100,
    verbose= 0
  )
  pred_matrix[, b] <- predict(model_b, newdata = as.matrix(X_sample))
}

pred_median <- apply(pred_matrix, 1, median)
pred_ci_low <- apply(pred_matrix, 1, quantile, probs = 0.025)
pred_ci_high<- apply(pred_matrix, 1, quantile, probs = 0.975)

df_uncertainty <- data.frame(
  Observed  = y_sample,
  Predicted = pred_median,
  CI_Lower  = pred_ci_low,
  CI_Upper  = pred_ci_high
) %>%
  mutate(
    CI_Width   = CI_Upper - CI_Lower,
    Trend_Class= case_when(
      Observed < -5 ~ "Strong Decline",
      Observed < 0  ~ "Moderate Decline",
      Observed < 5  ~ "Moderate Increase",
      TRUE          ~ "Strong Increase"
    )
  )

write.csv(df_uncertainty, file.path(out_dir, "xgb_uncertainty_sample.csv"), row.names = FALSE)

p1 <- ggplot(df_uncertainty, aes(Observed, Predicted)) +
  geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), width = 0.2, alpha = 0.3) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  theme_minimal() +
  labs(x = "Observed RSEI Trend", y = "Predicted RSEI Trend (±95% CI)")

p2 <- ggplot(df_uncertainty, aes(Trend_Class, CI_Width)) +
  geom_violin(alpha = 0.4, trim = TRUE) +
  geom_boxplot(width = 0.2, outlier.size = 0.6, fill = "white") +
  theme_minimal() +
  labs(x = "RSEI Trend Class", y = "95% CI Width")

combined_plot <- p1 / p2 + plot_layout(heights = c(1, 1))
ggsave(file.path(fig_dir, "uncertainty_combined.png"), combined_plot, width = 10, height = 10, dpi = 500, units = "in", bg = "white")
