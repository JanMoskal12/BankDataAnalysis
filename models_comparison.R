
library(tidymodels)

data <- readRDS("Dane_czyste.rds")

set.seed(2000)


data_split <- initial_split(data, 0.85)
train <- training(data_split)
test_data <- testing(data_split)


rf_fit <- readRDS("modele/wf_rf_best_fit.rds")

pred <- predict(rf_fit, new_data=test_data)
pred <- cbind(test_data, pred)
#conf_mat_rf <- conf_mat(pred, truth=Machine_failure, estimate=.pred_class)
met <- metric_set(f_meas, accuracy, sensitivity, specificity, recall, j_index)
rf_metrics <- met(data=pred, truth=y, estimate=.pred_class)
rf_metrics <- rf_metrics[,3]
colnames(rf_metrics) <- c("Random Forest")
rf_metrics


metric_names <- matrix(c("F1 score", "Accuracy", "Sensitivity", "Specificity", "Recall", "J index"), ncol=1)
colnames(metric_names) <- c("Metric")
metric_names <- as_tibble(metric_names)



svm_fit <- readRDS("modele/wf_best_svm_fit.rds")

pred <- predict(svm_fit, new_data=test_data)
pred <- cbind(test_data, pred)
svm_metrics <- met(data=pred, truth=y, estimate=.pred_class)
svm_metrics <- svm_metrics[,3]
colnames(svm_metrics) <- c("SVM")
svm_metrics

knn_fit <- readRDS("modele/wf_knn_best_fit.rds")

pred <- predict(knn_fit, new_data=test_data)
pred <- cbind(test_data, pred)
conf_mat_rf <- conf_mat(pred, truth=y, estimate=.pred_class)
knn_metrics <- met(data=pred, truth=y, estimate=.pred_class)
knn_metrics <- knn_metrics[,3]
colnames(knn_metrics) <- c("KNN")
knn_metrics


tree_fit <- readRDS("modele/wf_tree_best_fit.rds")

pred <- predict(tree_fit, new_data=test_data)
pred <- cbind(test_data, pred)
tree_metrics <- met(data=pred, truth=y, estimate=.pred_class)
tree_metrics <- tree_metrics[,3]
colnames(tree_metrics) <- c("Decision Tree")
tree_metrics

log_fit <- readRDS("modele/wfbest_regresja_fit.rds")

pred <- predict(log_fit, new_data=test_data)
pred <- cbind(test_data, pred)
log_metrics <- met(data=pred, truth=y, estimate=.pred_class)
log_metrics <- log_metrics[,3]
colnames(log_metrics) <- c("Logistic Regression")
log_metrics


boost_fit <- readRDS("modele/wfbest_boosting_fit.rds")

pred <- predict(boost_fit, new_data=test_data)
pred <- cbind(test_data, pred)
boost_metrics <- met(data=pred, truth=y, estimate=.pred_class)
boost_metrics <- boost_metrics[,3]
colnames(boost_metrics) <- c("Boosting")
boost_metrics

summary_table <- metric_names |> 
  cbind(rf_metrics, svm_metrics, knn_metrics, tree_metrics, log_metrics, boost_metrics)



summary_table

save(summary_table, file="wykresy/model_summary_table.rda")





rf_pred <- predict(rf_fit, new_data=test_data, type = "prob") %>% 
  bind_cols(test_data)
rf_roc <- rf_pred |>  
  roc_curve(y, .pred_0)

svm_pred <- predict(svm_fit, new_data=test_data, type = "prob") %>% 
  bind_cols(test_data)
svm_roc <- svm_pred |>  
  roc_curve(y, .pred_0)

knn_pred <- predict(knn_fit, new_data=test_data, type = "prob") %>% 
  bind_cols(test_data)
knn_roc <- knn_pred |>  
  roc_curve(y, .pred_0)

tree_pred <- predict(tree_fit, new_data=test_data, type = "prob") %>% 
  bind_cols(test_data)
tree_roc <- tree_pred |> 
  roc_curve(y, .pred_0)

log_pred <- predict(log_fit, new_data=test_data, type = "prob") %>% 
  bind_cols(test_data)
log_roc <- log_pred |> 
  roc_curve(y, .pred_0)

boost_pred <- predict(boost_fit, new_data=test_data, type = "prob") %>% 
  bind_cols(test_data)
boost_roc <- boost_pred |> 
  roc_curve(y, .pred_0)


curve <- ggplot()+
  geom_path(data=rf_roc,aes(x = 1 - specificity, y = sensitivity, col="darkblue"))+
  geom_path(data=svm_roc,aes(x = 1 - specificity, y = sensitivity ,col="#377eb8"))+
  geom_path(data=knn_roc,aes(x = 1 - specificity, y = sensitivity, col="#4daf4a"))+
  geom_path(data=tree_roc,aes(x = 1 - specificity, y = sensitivity, col="#984ea3"))+
  geom_path(data=log_roc, aes(x = 1 - specificity, y = sensitivity, col="#ff7f00"))+
  geom_path(data=boost_roc, aes(x = 1 - specificity, y = sensitivity, col="#ffff33"))+
  scale_color_identity(name = "Model",
                       breaks = c("darkblue", "#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33"),
                       labels = c("Random Forest","Support Vector Machines","K-nearest Neighbors", "Decision Tree","Logistic Regression","XGBoost"),
                       guide = "legend")+
  theme_bw()

curve

save(curve, file="wykresy/curve.rda")

