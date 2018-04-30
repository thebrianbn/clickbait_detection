library(tidytext)
library(tm)
library(caret)
library(kernlab)
library(e1071)
library(jsonlite)
library(stringr)
library(corpus)
library(extraTrees)
library(randomForest)

setwd("C:/Users/Jordan/Desktop/DS340/new_works")

svms = function(){

load("postText.Rdata")  ##pre processed data, to make life easier
testing_clickbait1 = postText[[2]]$clickbait
validation_clickbait1 = postText[[3]]$clickbait

run_svm = function(cost1=1, tol1=.001, training_clickbait, testing_clickbait){
model <- svm(clickbait ~., data=training_clickbait, kernel = "linear", scale=FALSE, cost = cost1, tolerance = tol1)

prediction <- predict(model, testing_clickbait)

comparison <- table(testing_clickbait$clickbait, prediction, dnn=c("Actual", "Predicted"))


visual <- confusionMatrix(prediction, testing_clickbait$clickbait, positive="good")
print(visual)
return(list(prediction,model))
}

pTPred2 = run_svm(cost=.22, training_clickbait = postText[[1]], testing_clickbait = postText[[2]])
postText3 = postText[[3]]
rm(postText)

load("targetTitle.Rdata")  ##pre-processed data
tTPred2 = run_svm(cost=.25, training_clickbait = targetTitle[[1]], testing_clickbait = targetTitle[[2]])
targetTitle3 = targetTitle[[3]]
rm(targetTitle)

load("tKeywords.Rdata")
tKeywords = targetKeywords
KeyPred2 = run_svm(cost=.25,training_clickbait = tKeywords[[1]], testing_clickbait = tKeywords[[2]])
tKeywords3 = tKeywords[[3]]
rm(tKeywords)

#allPred2 = run_svm(cost=.25,training_clickbait=training_clickbait_all,testing_clickbait=testing_clickbait_all)
#validation = merge(x,y,by="id")

tTModel = tTPred2[[2]]
tTPred = tTPred2[[1]]

pTModel = pTPred2[[2]]
pTPred = pTPred2[[1]]

keyModel = KeyPred2[[2]]
keyPred = KeyPred2[[1]]

#allModel = allPred2[[2]]
#allPred = allPred2[[1]]




##Stacking

##loading in logistic stuff

source('C:/Users/Jordan/Desktop/DS340/logis/logistic.R')

jul = logisp()


##brian's stuff

setwd("C:/Users/Jordan/Downloads")
brian5 = read.csv("predictions_21.csv")
brian_preds = sapply(brian5$results,function(x){ifelse(x==0,2,1)})
brian_preds1 = brian_preds[1:2500]

predDF = data.frame( as.numeric(tTPred),as.numeric(pTPred),jul[[1]]$tPlogPred, brian_preds1,stringsAsFactors = F)


modelET = extraTrees(as.matrix(predDF),testing_clickbait1)



testPredtT <- predict(tTModel, newdata = targetTitle3)
testPredpT <- predict(pTModel, newdata = postText3)
testKeyPred = predict(keyModel, newdata = tKeywords3)
logPredtP = jul[[2]]
brian_preds2 = brian_preds[2501:5000]


##adjusting for implementation


#testAllPred = predict(allModel,newdata = validation_all)
testPredLevelOne <- data.frame( testPredtT,testPredpT, logPredtP,brian_preds2,stringsAsFactors = F)
bs = testPredLevelOne
names(bs) = c("tT","pT","key")
bs$tT = as.numeric(bs$tT)
bs$pT = as.numeric(bs$pT)
bs$key = as.numeric(bs$key)
#bs$all = as.numeric(bs$all)
bs = as.matrix(bs)
combPred <- predict(modelET,bs)

print(confusionMatrix(combPred, validation_clickbait1))

return(list(testPredLevelOne,combPred,validation_clickbait1))

#return(confusionMatrix(combPred,validation_clickbait1))
}

check = svms()
