library(tidytext)
library(tm)
library(caret)
library(kernlab)
library(e1071)
library(jsonlite)
library(stringr)
library(corpus)
library(gbm)

setwd("C:/Users/Jordan/Desktop/DS340/big")
#if(FALSE){
train = read.csv("good_data.csv")
#train = train[1:10000,]
train$clickbait = as.factor(ifelse(train$truthClass=="clickbait","clickbait","good"))

f = function(x){paste(unlist(x),collapse= " ")}
train$targetParagraphs = sapply(train$targetParagraphs,f)
#train$targetCaptions = sapply(train$targetCaptions,f)
train$targetKeywords = sapply(train$targetKeywords,f)
train$postText = unlist(train$postText)

postText = train[,c(str1,"clickbait","id")]
names(postText) = c("postText","clickbait","id")
new = postText
new$postText = str_replace_all(new$postText,"[^[:graph:]]", " ")


#new = postText[,c(2,1)]
new$clickbait = as.factor(new$clickbait)
new$postText = as.character(new$postText)
new$postText = iconv(new$postText,"latin1","ASCII",sub="")
new$postText = tolower(new$postText)



lp = new$postText

temp = text_tokens(lp, stemmer = "en")
new$postText = sapply(temp,paste,collapse = " ")
new$postText = removePunctuation(new$postText)
new$postText = gsub(" \\d+ "," [n] ",new$postText)
new$postText = stripWhitespace(new$postText)
##########################################################################

# names(new)
# new[1,]

save(new,file="targetParagraphs_logis.Rdata")



############CLEANING############################################
##removing weird characters, whitespace, punctuation, stopwords, and making everything lowercase.
##leaving numbers, as those are likely important in clickbait
new2 = Corpus(VectorSource(new$postText))
#new2 = tm_map(new2, function(x) iconv(enc2utf8(x), sub = "byte")) ##this removes all weird characters
#new2 = tm_map(new2, content_transformer(tolower))
#new2 = tm_map(new2, removeWords, stopwords())
#new2 = tm_map(new2, stripWhitespace)
#new2 = tm_map(new2, removePunctuation)

##Creating Document Term Matrices with tm as part of cleaning
dtm = DocumentTermMatrix(new2)
features = findFreqTerms(dtm, 10)
dtm2 = DocumentTermMatrix(new2, list(global = c(2, Inf),
                                     dictionary = features))

#####dividing training and testing data ####
##when ran on supercomputer later, will be able to handle more training/testing data
 train1 = new[1:14000,]   ##for now, taking the last 100 records of our dataset to be our testing data.  When we run this on XSEDE, we'll have two separate sets for training and testing
 test1 = new[14001:16500,]
 validation1 = new[16501:19000,]


 train2 = new2[1:14000]
 test2 = new2[14001:16500]
 validation2 = new2[16501:19000]

#######DTM --> Data Frame######################

dict2 = findFreqTerms(dtm2, lowfreq=10)
training_clickbait = DocumentTermMatrix(train2, list(dictionary=dict2))
testing_clickbait = DocumentTermMatrix(test2,list(dictionary=dict2))
validation = DocumentTermMatrix(validation2,list(dictionary=dict2))

##allows for DTM --> data frame
training_clickbait <- as.data.frame(apply(training_clickbait,MARGIN=2, FUN=function(x){x = ifelse(x>0,1,0)}))
testing_clickbait <- as.data.frame(apply(testing_clickbait,MARGIN=2, FUN=function(x){x=ifelse(x>0,1,0)}))
validation = as.data.frame(apply(validation,MARGIN=2,FUN=function(x){x = ifelse(x>0,1,0)}))

#########preparing for svm function#####################

training_clickbait <- cbind(clickbait=factor(train1$clickbait), training_clickbait)
testing_clickbait <- cbind(clickbait=factor(test1$clickbait), testing_clickbait)
validation = cbind(clickbait = factor(validation1$clickbait),validation)
#}

#training_clickbait_postText = training_clickbait
#testing_clickbait_postText = testing_clickbait
#validation_postText = validation
postText = list(training_clickbait,testing_clickbait,validation)
save(postText,file="postText.Rdata")

#postText = list(training_clickbait_postText,testing_clickbait_postText,validation_postText)

#training_clickbait_targetTitle = training_clickbait
#testing_clickbait_targetTitle = testing_clickbait
#validation_targetTitle = validation

targetTitle = list(training_clickbait,testing_clickbait,validation)
save(targetTitle,file="targetTitle.Rdata")

targetKeywords = list(training_clickbait,testing_clickbait,validation)
save(targetKeywords,file="targetKeywords.Rdata")

#training_clickbait_tKeywords = training_clickbait
to_save = training_clickbait
#testing_clickbait_tKeywords = testing_clickbait
#validation_tKeywords = validation

#tKeywords = list(training_clickbait_tKeywords, testing_clickbait_tKeywords, validation_tKeywords)
save(to_save,file="targetKeywords_corpus")
#training#_clickbait_all = training_clickbait
#testing_clickbait_all = testing_clickbait
#validation_all = validation


