library(daltoolbox)

data(Boston)
data <- Boston

preproc <- minmax()
preproc <- fit(preproc, data)
data <- transform(preproc, data)

sample <- sample_random()
tt <- train_test(sample, data)
train <- tt$train
test <- tt$test

library(reticulate)
source_python('pytorch/python_basics.py')

savedf(as.data.frame(train), 'pytorch/data_reg_train.csv')
savedf(as.data.frame(test), 'pytorch/data_reg_test.csv')
