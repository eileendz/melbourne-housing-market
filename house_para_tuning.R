library(tfruns)

runs <- tuning_run("house.R", flags = list(
  dense_unit1 = c(2048, 1024, 512), #c(512, 256, 128, 64)
  dense_unit2 = c(1024, 512, 256), 
  dense_unit3 = c(512, 256, 128), 
  dropout = c(0.2), # dropout rate of 0.2 seems to be better than 0.4
  learning_rate = c(1e-3), # could try if 1e-4 or 1e-5 would work better but need more epoch
  batch_size = c(64, 128,256, 512), #, 
  activation = c("relu") # LeakyReLU, , "tanh"
))

# runs <- tuning_run("house.R", flags = list(
#   dropout = c(0.2),
#   learning_rate = c(1e-3), 
#   epoch = c(30), # could optimise with better grid value
#   batch_size = c(20) # c(16, 32, 64)
# ))


run_result <- ls_runs(latest_n = 138) %>% unnest(metrics) %>% 
  dplyr::arrange(metric_val_loss) %>% 
  select(metric_loss, metric_acc, metric_val_loss, metric_val_acc, flag_dropout, flag_dense_unit1, flag_dense_unit2,
         flag_dense_unit3, flag_batch_size, flag_activation, flag_learning_rate, run_dir)

view_run(run_result$run_dir[1])

# more work could be done if we have a better computer
# increase patience to 20 so we can try learning rate at 1e-4 or 1e-5
# play around with different activation function: tanh or leaky relu - this would affect dropout tbh
