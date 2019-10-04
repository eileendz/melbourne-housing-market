library(tidyverse)


# load data & factorisation
house_less_raw<- read_csv("data/MELBOURNE_HOUSE_PRICES_LESS.csv") %>% 
  mutate(suburb = as.factor(Suburb), 
         address = as.factor(Address),
         type = as.factor(Type), 
         method = as.factor(Method), 
         date = as.Date(Date,"%d/%m/%Y"),
         postcode = as.factor(Postcode), 
         regionname = as.factor(Regionname),
         councilarea = as.factor(CouncilArea), 
         price = Price/ 1e3)

# drop missing house price, assign category (1-10) to price based on quantile. 
# Then our problem can be re-phrased into: predict which category will the price 
# falls into, which is a classification problem!
house_less<- house_less_raw %>% 
  filter(!is.na(price)) %>% 
  mutate(Price_class = findInterval(price, quantile(price, probs=0:10/10)))

# no missings!
naniar::vis_miss(house_less, warn_large_data = FALSE)

# For the nn, we could do something like "self-organizing map (SOM)" or "Learning vector quantization" (LVQ), 
# which is popular for multi-label classification. Multi-Task Learning may also
# work. Here are some papers: 

# https://link.springer.com/content/pdf/10.1023%2FA%3A1023977111302.pdf
# https://arxiv.org/pdf/1901.01774.pdf

library(kohonen)
som_grid <- somgrid(xdim = 20, ydim=20, topo="hexagonal")



