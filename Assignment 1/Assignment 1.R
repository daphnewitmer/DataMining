library(ggplot2)
library(ggpubr)
library(caret) # for 10-fold crossvalidation
library(klaR)
library(rlang)
library(labelled)

data = read.csv("C:/Users/Kristian/Documents/VU/Data Mining/Assignment 1/ODI-2021.csv",
                stringsAsFactors = F)
colnames(data) = c("time", "program", "mlcourse", "infcourse", "statcourse", "datacourse",
                   "gender", "chocolate", "birthday", "neighbors", "standup",
                   "stress", "localDM", "number", "bedtime", "goodday1", "goodday2")

# Convert 0,1/mu, sigma etc. to yes/no
data[which(data$infcourse==1), "infcourse"] = "yes"
data[which(data$infcourse==0), "infcourse"] = "no"
data[which(data$statcourse=="mu"), "statcourse"] = "yes"
data[which(data$statcourse=="sigma"), "statcourse"] = "no"
data[which(data$datacourse=="ja"), "datacourse"] = "yes"
data[which(data$datacourse=="nee"), "datacourse"] = "no"

data[which(data$chocolate=="I have no idea what you are talking about"), "chocolate"] = "no idea"

male = data[data$gender=="male", ]
female = data[data$gender=="female", ]

mlplotm = ggplot(male, aes(x=mlcourse)) + geom_bar() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
  xlab("ML") + ylab("Frequency")
infplotm = ggplot(male, aes(x=infcourse)) + geom_bar() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Inf. retrieval") + ylab("Frequency")
statplotm = ggplot(male, aes(x=statcourse)) + geom_bar() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Statistics") + ylab("Frequency")
dataplotm = ggplot(male, aes(x=datacourse)) + geom_bar() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
  xlab("Databases") + ylab("Frequency")
mlplotf = ggplot(female, aes(x=mlcourse)) + geom_bar() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
  xlab("ML") + ylab("Frequency")
infplotf = ggplot(female, aes(x=infcourse)) + geom_bar() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Inf. retrieval") + ylab("Frequency")
statplotf = ggplot(female, aes(x=statcourse)) + geom_bar() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Statistics") + ylab("Frequency")
dataplotf = ggplot(female, aes(x=datacourse)) + geom_bar() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
  xlab("Databases") + ylab("Frequency")
combineplotmf = ggarrange(mlplotm,infplotm,statplotm,dataplotm,mlplotf,infplotf,statplotf,dataplotf,
                        labels = c("A", "B", "C", "D", "E", "F", "G", "H"), nrow = 2, ncol = 4, label.y = 0.3)
ggsave("C:/Users/Kristian/Documents/VU/Data Mining/Assignment 1/courses_per_gender.pdf", width = 6, height = 3.5)

               
# "Random" number
data$number = as.numeric(data$number)
sum(data$number>0 & data$number < 11,na.rm=T)
binom.test(44, 313, 1/10, alternative = "greater")

# Bedtimes
beddata = data[nchar(data$bedtime)==5, ]
beddata[which(beddata$bedtime=="11 pm"), "bedtime" ] = "23.00"
beddata = beddata[-which(beddata$bedtime=="What?" | beddata$bedtime == "00000"), ] 
beddata$bedtime = gsub(":", ".", beddata$bedtime)
beddata$bedtime = gsub(";", ".", beddata$bedtime)
beddata$bedtime = gsub("-", ".", beddata$bedtime)
beddata$bedtime = gsub(",", ".", beddata$bedtime)
beddata$bedtime = gsub("h", ".", beddata$bedtime)
beddata$bedtime = as.numeric(beddata$bedtime)

beddata_men = setNames(data.frame(beddata[beddata$gender=="male", "bedtime"]), "bedtime")
beddata_women = setNames(data.frame(beddata[beddata$gender=="female", "bedtime"]), "bedtime")

bedtime_men = ggplot(beddata_men, aes(x=bedtime)) + geom_density() + scale_x_continuous(breaks=seq(0,24,6)) + ylim(0, 0.06) +
  xlab("Time (hours)") + ylab("Frequency"); bedtime_men
bedtime_women = ggplot(beddata_women, aes(x=bedtime)) + geom_density() + scale_x_continuous(breaks=seq(0,24,6)) + ylim(0, 0.06) + 
  xlab("Time (hours)") + ylab("Frequency"); bedtime_women
combine_bedtime = ggarrange(bedtime_men, bedtime_women, labels = c("A", "B"), ncol = 2)
ggsave("C:/Users/Kristian/Documents/VU/Data Mining/Assignment 1/combine_bedtime.pdf", width = 4, height = 2)

  #bedtime_men = ggplot(beddata_men, aes(x=bedtime)) + geom_histogram(); bedtime_men
#bedtime_women = ggplot(beddata_women, aes(x=bedtime)) + geom_histogram(binwidth = 1); bedtime_women
#combine_bedtime = ggarrange(bedtime_men, bedtime_women, labels = c("A", "B"), ncol = 2)


# Task 1B
# Naive Bayes

task1bdata = data
  task1bdata = data[data$gender!="unknown", ]
  task1bdata = task1bdata[-sample(which(task1bdata$gender=="male"), size = 103), ]
task1bdata[sapply(data, is.character)] = lapply(task1bdata[sapply(task1bdata, is.character)], as.factor)

indices = sample(nrow(task1bdata), floor(nrow(task1bdata)*0.7)) # Split 70/30
train = task1bdata[indices,]
test = task1bdata[-indices,]

xTrain = train[, c("mlcourse", "infcourse", "statcourse", "datacourse")]
yTrain = train$gender 

xTest = test[, c("mlcourse", "infcourse", "statcourse", "datacourse")]
yTest = test$gender

model = train(xTrain,yTrain,'nb',trControl=trainControl(method='cv',number=10)); model

predict(model$finalModel,xTest)$class
prop.table(table(predict(model$finalModel,xTest)$class,yTest))

# K-nearest neighbors
task1bdata = data
  task1bdata = data[data$gender!="unknown", ]
  task1bdata = task1bdata[-sample(which(task1bdata$gender=="male"), size = 103), ]
task1bdata[task1bdata=="yes"] = 1
task1bdata[task1bdata=="no"] = 0
task1bdata[task1bdata=="male"] = 1
task1bdata[task1bdata=="female"] = 0
task1bdata = task1bdata[, c("mlcourse", "infcourse", "statcourse", "datacourse", "gender")]

indices = sample(nrow(task1bdata), floor(nrow(task1bdata)*0.7)) # Split 70/30
train = task1bdata[indices,]
test = task1bdata[-indices,]

fit = train(gender ~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:10),
             trControl  = trainControl(method  = "cv", number  = 10),
             metric     = "Accuracy",
             preProcess = c("center","scale"),
             data       = train)

test$gender = as.factor(test$gender)
predicting = predict(fit, newdata = test); predicting 
confusionMatrix(predicting,  test$gender)



