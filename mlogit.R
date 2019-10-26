library(nnet)
load("data/house_c.rda")
load("data/train.rda")
load("data/test.rda")

mlogit <- multinom(price_class ~ rooms + type + method + postcode + distance + council_area +
           region_name + property_count, data=train, MaxNWts = 3000)

pred <- predict (mlogit, test, "class")

pred_mnl <- table(pred, test$price_class)
save(pred_mnl, file = "data/pred_mnl")

mean(as.character(pred) != as.character(test$price_class))
