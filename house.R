library(tidyverse)


# load data & factorisation
house_less_raw<- read_csv("data/MELBOURNE_HOUSE_PRICES_LESS.csv") %>% 
  mutate(Suburb = as.factor(Suburb), 
         Address = as.factor(Address),
         Type = as.factor(Type), 
         Method = as.factor(Method), 
         Date = as.Date(Date,"%d/%m/%Y"),
         Postcode = as.factor(Postcode), 
         Regionname = as.factor(Regionname),
         CouncilArea = as.factor(CouncilArea)) 

# drop missing house price, assign category (1-10) to price based on quantile. 
# Then our problem can be re-phrased into: predict which category will the price 
# falls into, which is a classification problem!
house_less<- house_less_raw %>% 
  filter(!is.na(Price)) %>% 
  mutate(Price_class = findInterval(Price, quantile(Price, probs=0:10/10)))

# no missings!
naniar::vis_miss(house_less)

# For the nn, we could do something like "self-organizing map (SOM)" or "Learning vector quantization" (LVQ), 
# which is popular for multi-label classification. Multi-Task Learning may also
# work. Here are some papers: 

# https://link.springer.com/content/pdf/10.1023%2FA%3A1023977111302.pdf
# https://arxiv.org/pdf/1901.01774.pdf

  


