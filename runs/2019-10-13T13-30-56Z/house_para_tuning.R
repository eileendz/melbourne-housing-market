library(tfruns)

runs <- tuning_run("house.R", flags = list(
  dense_unit1 = c(512, 256, 128, 64), 
  dense_unit2 = c(256, 128, 64, 32), 
  dropout = c(0.2, 0.3, 0.4), 
  learning_rate = c(1e-3, 1e-4, 1e-5), # could try if 1e-4 or 1e-5 would work better but need more epoch
  epoch = c(30, 50, 100, 200), # could optimise with better grid value
  batch_size = c(16, 32,64, 128) # c(16, 32, 64)
))

# runs <- tuning_run("house.R", flags = list(
#   dropout = c(0.2),
#   learning_rate = c(1e-3), 
#   epoch = c(30), # could optimise with better grid value
#   batch_size = c(20) # c(16, 32, 64)
# ))


run_result <- ls_runs() %>% unnest(metrics) %>% 
  dplyr::arrange(-metric_acc) %>% 
  select(metric_loss, metric_acc, flag_dropout, flag_density_unit1, flag_density_unit2,
         flag_epoch, flag_batch_size, flag_learning_rate, run_dir)

view_run(run_result$run_dir[1])
