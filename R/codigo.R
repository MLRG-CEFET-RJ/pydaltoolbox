library(daltoolbox)
library(harbinger)

data(har_examples_multi)

#Using the time series 9
dataset <- har_examples_multi[[1]]
head(dataset)

plot_ts(x = 1:length(dataset$x), y = dataset$x)
plot_ts(x = 1:length(dataset$y), y = dataset$y)
dataset$event <- NULL

norm <- minmax()
norm <- fit(norm, dataset)
dataset <- transform(norm, dataset)
summary(dataset)
write.table(dataset, file="dataset.csv", row.names=FALSE, quote = FALSE, sep = ",")

plot_ts(x = 1:length(dataset$x), y = dataset$x)
plot_ts(x = 1:length(dataset$y), y = dataset$y)


datax <- as.data.frame(ts_data(dataset$x, 5))
colnames(datax) <- c("x4", "x3", "x2", "x1", "x0")
write.table(datax, file="datax.csv", row.names=FALSE, quote = FALSE, sep = ",")


datay <- as.data.frame(ts_data(dataset$y, 5))
colnames(datay) <- c("y4", "y3", "y2", "y1", "y0")
write.table(datay, file="datay.csv", row.names=FALSE, quote = FALSE, sep = ",")


dataxy <- cbind(datax, datay)