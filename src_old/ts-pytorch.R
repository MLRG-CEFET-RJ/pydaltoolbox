setwd("~/develop/pytorch")

source("myBasic.R")
source("myPreprocessing.R")

library(reticulate)
source_python('ts-pytorch.py')


t <- 0:199
x <- sin(t)

preproc <- ts_gminmax()

sw_size <- 5
ts <- ts_data(x, sw_size)

test_size <- 20
samp <- ts_sample(ts, test_size)
train <- samp$train
test_size <- 10
samp <- ts_sample(samp$test, test_size)
valid <- samp$train
test <- samp$test


io_train <- ts_projection(train)
io_valid <- ts_projection(valid)
io_test <- ts_projection(test)

savedf(as.data.frame(train), 'train_sin.csv')
savedf(as.data.frame(valid), 'val_sin.csv')
savedf(as.data.frame(test), 'test_sin.csv')

if (TRUE) {
  model <- create_model(4, 3)
  model <- train_pytorch(model, as.data.frame(train))
  savemodel(model, "model.pt")
}

if (TRUE) {
  model <- loadmodel("model.pt")
  mytest <- as.data.frame(io_test$input)
  mytest$t0 <- 0
  pytest <- predict_pytorch(model, mytest)
  plot(pytest)
  #print(model)
}


