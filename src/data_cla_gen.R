library(daltoolbox)

data(iris)
data <- iris

preproc <- minmax()
preproc <- fit(preproc, data)
data <- transform(preproc, data)

cm <- categ_mapping("Species")
data <- cbind(data, transform(cm, data))
data$Species <- NULL

sample <- sample_random()
tt <- train_test(sample, data)
train <- tt$train
test <- tt$test

library(reticulate)
source_python('pytorch/python_basics.py')

savedf(as.data.frame(train), 'pytorch/data_cla_train.csv')
savedf(as.data.frame(test), 'pytorch/data_cla_test.csv')


