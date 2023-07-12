library(daltoolbox)
library(harbinger)

#loading the example database
data(har_examples)

#Using the time series 1 
dataset <- har_examples[[17]]

sw_size <- 4
ts <- ts_data(dataset$serie, sw_size)

preproc <- ts_norm_gminmax()
preproc <- fit(preproc, ts)
ts <- transform(preproc, ts)
ts <- as.data.frame(ts)
ts$event <- as.integer(dataset$event[sw_size:length(dataset$event)])

samp <- ts_sample(ts, test_size = 30)
train <- samp$train
test <- samp$test

library(reticulate)
source_python('pytorch/python_basics.py')

savedf(as.data.frame(train), 'pytorch/data_tscla_train.csv')
savedf(as.data.frame(test), 'pytorch/data_tscla_test.csv')

