library(tidyverse)
library(keras)
library(rsample)
library(recipes)

# load data & cleaning
house_less<- read_csv("data/MELBOURNE_HOUSE_PRICES_LESS.csv") %>% 
  janitor::clean_names() %>% 
  mutate(suburb = as_factor(suburb), 
         address = as_factor(address),
         type = as_factor(type), 
         method = as_factor(method), 
         postcode = as_factor(postcode), 
         region_name = as_factor(regionname),
         property_count = propertycount,
         council_area = as_factor(council_area), 
         price = price/ 1e3) %>% 
  # to deal with address maybe one hot encoding - for simplicity, not included
  # dont think seller would affect house price so not included
  select(-c(address, seller_g, propertycount, regionname)) %>% 
  # drop the observation if no price information
  filter(!is.na(price)) 

# the data is right skewed! 
house_less %>% ggplot(aes(x = price)) + geom_histogram()

house_c <- house_less %>% 
  # assign category (1-10) to price based on range (0, 100], (100, 300], (300, 500] ... (1500, 1700], (1700, $\infty$)
  # notice that the price in this thousand dollars - see line 17
# quantile(house_less$price,probs = seq(0, 1, 0.1))
# c(0, seq(1e2, 18e2, 2e2), max(house_less$price)) # notice inbalanced data - thus using loss function rather than accuracy as measuring criteria
  mutate(price_class = cut(house_less$price, breaks = c(0, seq(1e2, 18e2, 2e2), max(house_less$price)), label = FALSE)) %>% 
  select(-c(price, suburb, date)) # suburb and postcode are giving the same information maybe take only one?

# Then our problem can be re-phrased into: predict which category will the price 
# falls into, which is a classification problem!

# no missings!
#naniar::vis_miss(house_less, warn_large_data = FALSE)


# resampling: train test  split: 0.9, 0.1
set.seed(92472)
house_split <- initial_split(house_c, prop = 0.9)  
train <- house_split %>% training()
test <- house_split %>% testing()

# pre-processing data 
rec_obj <- train %>% 
  recipe(price_class ~.) %>% 
  step_center(all_numeric(),-all_outcomes()) %>% 
  step_scale(all_numeric(), -all_outcomes()) %>% # rescale numerical values
  step_dummy(all_nominal(), -all_outcomes()) %>% # one hot encoding for factors 
  prep()

train_x <- bake(rec_obj,new_data = train) %>% select(-price_class) 
train_x <- train_x %>% as.matrix()
train_y <- to_categorical(pull(train, price_class)-1, 10)
test_x <- bake(rec_obj, new_data = test) %>% select(-price_class)
test_y <- to_categorical(pull(test, price_class)-1, 10)

# NN model
# dont really find a pre-trained architecture for house price forcasting, although there are articles on using ANN to predict
# thus we build our own from scratch. 

# architecture: 
# currently have two hidden layers - definitely could add more!
# batch normalisation is necessary because relu is a not a zero centered activation function, including batch normalisation can speed up the convergence 
# activation function - middle: "relu" - it doesnt go flat near -1 and 1 but we need to deal with its not zero centered characteristics: batch normalisation (other possible choose includes tanh)
# activation function - final: "softmax" at final layer because we are doing multi-level classification
# dropout to avoid overfitting

# compilation: 
# loss function: "categorical_crossentropy" because multi-class classification
# optimiser: see assgin 3: adam may not different from rmsprop much, could tune the learning rate
# metrics: accuracy

FLAGS <- flags(
  flag_integer("dense_unit1", 1024),
  flag_numeric("dropout", 0.2), 
  flag_integer("dense_unit2", 512),
  flag_integer("dense_unit3", 128),
  flag_numeric("learning_rate", 1e4), 
  flag_integer("epoch", 500), 
  flag_integer("batch_size", 32), 
  flag_string("activation", "relu")
)
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 5) #restore_best_weights = TRUE


k_clear_session()
model <- keras_model_sequential() %>% 
  layer_dense(units = FLAGS$dense_unit1, activation = FLAGS$activation, input_shape = 277) %>%  # input_shape is the dimension of the input EXCLUDING THE SAMPLE AXIS!and dont put c(277) - for some reason it doesnt work!
  layer_dropout(rate = FLAGS$dropout) %>% 
  layer_batch_normalization() %>% 
  layer_dense(units = FLAGS$dense_unit2, activation = FLAGS$activation) %>% 
  layer_dropout(rate = FLAGS$dropout) %>% 
  layer_batch_normalization() %>% 
  layer_dense(units = FLAGS$dense_unit3, activation = FLAGS$activation) %>% 
  layer_dropout(rate = FLAGS$dropout) %>% 
  layer_batch_normalization() %>% 
  layer_dense(units = 10, activation = "softmax") %>% 
  compile(optimizer = optimizer_adam(FLAGS$learning_rate), 
          loss = "categorical_crossentropy", 
          metrics = "acc")

# parameter tuning via cross validation

# cross validation
# k <- 3 # could use 5 or 10 tho takes more time 
# set.seed(92472)
# ind <- sample(1: nrow(train_x))
# fold <- cut(ind, breaks = k, label = FALSE)
# score <- c()
# 
# cv_function <- function(i){
#   result <- tibble::as_tibble()
#   epoch_num <-  FLAGS$epoch
#   val_ind <- which(fold == i)
#   validation_x <- train_x[val_ind,] 
#   validation_y <- train_y[val_ind,]
#   train_x_cv <- train_x[-val_ind,]
#   train_y_cv <- train_y[-val_ind,]
#   
#   history <- model %>% fit(train_x_cv, train_y_cv, 
#                            validation_adta = list(validation_x, validation_y), 
#                            epoch =epoch_num , batch_size = FLAGS$batch_size)
#   
#   ind_result <- result %>% bind_rows(tibble::as_tibble(history))
#   return(ind_result)
# }
# 
# cv_result <- map_df(1:k, cv_function)
# 
# cv_result_tidy <- cv_result %>% filter(epoch == 2) %>% 
#   group_by(metric) %>% 
#   summarise(accuracy = mean(value))

# val_ind <- sample(1:nrow(train_x),size = floor(0.9*nrow(train_x))) # thus 0.81 as training, 0.9 as validation, 10% as testing
# x_validation <- train_x[val_ind,] 
# y_validation <- train_y[val_ind,]
# x_train <- train_x[-val_ind,]
# y_train <- train_y[-val_ind,]

history <- model %>% fit(train_x, train_y,
                         validation_split = 0.3,
                         epoch =FLAGS$epoch , batch_size = FLAGS$batch_size,
                         callbacks = early_stop)

ind_result <- tibble::as_tibble(history$metrics)






# For the nn, we could do something like "self-organizing map (SOM)" or "Learning vector quantization" (LVQ), 
# which is popular for multi-label classification. Multi-Task Learning may also
# work. Here are some papers: 

## maybe just stick to the ones taught in class???

