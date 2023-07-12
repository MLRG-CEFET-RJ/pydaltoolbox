library(daltoolbox)

data(sin_data)

sw_size <- 3
ts <- ts_data(sin_data, sw_size)

preproc <- ts_norm_gminmax()
preproc <- fit(preproc, ts)
ts <- transform(preproc, ts)


samp <- ts_sample(ts, test_size = 10)
train <- samp$train
test <- samp$test

library(reticulate)
source_python('pytorch/python_basics.py')

savedf(as.data.frame(train), 'pytorch/data_ts_train.csv')
savedf(as.data.frame(test), 'pytorch/data_ts_test.csv')
