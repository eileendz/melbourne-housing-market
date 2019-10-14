library(tfruns)

runs <- tuning_run("house.R", flags = list(
  dropout = c(0.2, 0.4), 
  epoch = c(2, 5), # c(30, 50, 100)
  batch_size = c(10, 20) # c(16, 32, 64)
))


run_result <- ls_runs() %>% unnest(metrics) %>% 
  dplyr::arrange(-metric_acc) %>% 
  select(metric_loss, metric_acc, flag_dropout, 
         flag_epoch, flag_batch_size, run_dir)

view_run(run_result$run_dir[1])
