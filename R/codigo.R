library(daltoolbox)
library(harbinger)

data(har_examples_multi)

#Using the time series 9
dataset <- har_examples_multi[[1]]
head(dataset)
dataset$event <- NULL

norm <- minmax()
norm <- fit(norm, dataset)
dataset <- transform(norm, dataset)
summary(dataset)
plot_ts(x = 1:length(dataset$x), y = dataset$x)
plot_ts(x = 1:length(dataset$y), y = dataset$y)

write.table(dataset, file="dataset.csv", row.names=FALSE, col.names = FALSE, quote = FALSE, sep = ",")


datax <- as.data.frame(ts_data(dataset$x, 5))
head(datax)
write.table(datax, file="datax.csv", row.names=FALSE, col.names = FALSE, quote = FALSE, sep = ",")


datay <- as.data.frame(ts_data(dataset$y, 5))
head(datay)
write.table(datay, file="datay.csv", row.names=FALSE, col.names = FALSE, quote = FALSE, sep = ",")


dataxy <- cbind(datax, datay)
head(dataxy)
write.table(dataxy, file="dataxy.csv", row.names=FALSE, col.names = FALSE, quote = FALSE, sep = ",")
