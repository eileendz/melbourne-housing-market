library(keras)

load("data/train_x.rda")
load("data/train_y.rda")
load("data/test_x.rda")
load("data/test_y.rda")

k_clear_session()
model <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", input_shape = 277,
              kernel_regularizer = regularizer_l2(l = 1e-3)) %>%  # input_shape is the dimension of the input EXCLUDING THE SAMPLE AXIS!and dont put c(277) - for some reason it doesnt work!
  layer_dropout(rate = 0.2) %>% 
  layer_batch_normalization() %>% 
  layer_dense(units = 64, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 1e-3)) %>% 
  layer_dropout(rate = 00.2) %>% 
  layer_batch_normalization() %>% 
  layer_dense(units = 10, activation = "softmax") %>% 
  compile(optimizer = optimizer_adam(1e-4), 
          loss = "categorical_crossentropy", 
          metrics = "acc")

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 5,
                                      restore_best_weights = TRUE) 

model %>% fit(train_x, train_y,
              validation_split = 0.3,
              epoch =100 , batch_size = 256,
              callbacks = early_stop)

eval <- model %>% evaluate(as.matrix(test_x), test_y)

predicted_class <- model %>% predict_classes(as.matrix(test_x))

