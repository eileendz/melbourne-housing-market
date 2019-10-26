library(tfruns)

runs <- tuning_run("house.R", flags = list(
  dense_unit1 = c(256, 128), #c(512, 256, 128, 64)
  dense_unit2 = c(128, 64),
  dropout = c(0.1, 0.2), # dropout rate of 0.2 seems to be better than 0.4
  batch_size = c(256, 512), #,
  activation = c("tanh") # LeakyReLU, , "tanh"
))

# may want to try if three layer is good but doubt about that

run_result <- ls_runs(latest_n = 108) %>% unnest(metrics) %>%
  dplyr::arrange(metric_val_loss) %>%
  select(metric_loss, metric_acc, metric_val_loss, metric_val_acc, flag_dropout,
         flag_batch_size, flag_activation, flag_dense_unit1, flag_dense_unit2 ,run_dir)


view_run("runs/2019-10-22T12-51-42Z") 

#1	1.2804	0.4975	1.2784	0.5043	0.2	256	tanh	128	64 not really much difference

