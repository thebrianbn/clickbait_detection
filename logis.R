library(tidytext)
library(tm)
library(caret)
library(kernlab)
library(e1071)
library(jsonlite)
library(stringr)
library(corpus)
library(gbm)
library(text2vec)
library(data.table)
library(magrittr)

setwd("C:/Users/Jordan/Desktop/DS340")
x = stream_in(file("big/instances.jsonl",open="r"))
y = stream_in(file("big/truth.jsonl",open="r"))
#if(FALSE){
train = merge(x,y,by="id")
#train = train[1:10000,]
train$clickbait = as.factor(ifelse(train$truthClass=="clickbait","clickbait","good"))

f = function(x){paste(unlist(x),collapse= " ")}
train$targetParagraphs = sapply(train$targetParagraphs,f)
#train$targetCaptions = sapply(train$targetCaptions,f)
train$targetKeywords = sapply(train$targetKeywords,f)
train$postText = unlist(train$postText)

postText = train[,c(str1,"clickbait")]
names(postText) = c("postText","clickbait")
postText$postText = str_replace_all(postText$postText,"[^[:graph:]]", " ")


new = postText[,c(2,1)]
new$clickbait = as.factor(new$clickbait)
new$postText = as.character(new$postText)
new$postText = iconv(new$postText,"latin1","ASCII",sub="")
new$postText = tolower(new$postText)
#new$postText = removeWords(new$postText, stopwords())
lp = new$postText

temp = text_tokens(lp, stemmer = "en")
new$postText = sapply(temp,paste,collapse = " ")
new$postText = removePunctuation(new$postText)
new$postText = gsub(" \\d+ "," [n] ",new$postText)
new$postText = stripWhitespace(new$postText)

train5 = new[1:18000,]
test5 = new[18001:19000,]

prep_fun = tolower
tok_fun = word_tokenizer

it_train = itoken(train5$postText, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = train5$id, 
                  progressbar = FALSE)
vocab = create_vocabulary(it_train)

vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

library(glmnet)
NFOLDS = 4
glmnet_classifier = cv.glmnet(x = dtm_train, y = train5[['clickbait']], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = NFOLDS,
                              thresh = 1e-3,
                              maxit = 1e3)

it_test = test5$postText %>% 
  prep_fun %>% tok_fun %>% 
  itoken(ids = test5$id, progressbar = FALSE)


dtm_test = create_dtm(it_test, vectorizer)

preds = predict(glmnet_classifier, dtm_test, type = 'response')[,1]
for(i in seq(1:length(preds))){
  preds[i] = ifelse(preds[i]>.5,2,1)
  
}
test5$clickbait = as.integer(test5$clickbait)
visual <- confusionMatrix(preds, test5$clickbait, positive="1")



head(preds)

