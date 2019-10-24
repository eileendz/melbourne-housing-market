library(nnet)
load("data/house_c.rda")
load("data/train.rda")
load("data/test.rda")

mlogit <- multinom(price_class ~ rooms + type + method + postcode + distance + council_area +
           region_name + property_count, data=train, MaxNWts = 3000)

pred <- predict (mlogit, test, "class")

table(pred, test$price_class)

mean(as.character(pred) != as.character(test$price_class))
