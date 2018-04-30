##The framework for this shiny application comes from http://deanattali.com/blog/building-shiny-apps-tutorial/, everything else comes from documentation and prior knowledge

library(shiny)
library(scales)
library(rsconnect)
ui <- fluidPage( ##establish the user interface
  titlePanel("Computational System for Clickbait Detection"),
  #first time using shiny, not sure how to get rid of this entirely
    mainPanel(
      h4( "The Tweet Text:", style = "font-weight:bold"),
      h2(textOutput("raw")),
      h4("Article Title:", style = "font-weight:bold"),
      h2(textOutput("raw2")),
      h2("Our Predictions:"),
      tableOutput("try1"))

    )
server <- function(input, output) {  ##pure R code
  
  #source('C:/Users/Jordan/Desktop/DS340/new_works/newer.R')
  setwd("C:/Users/Jordan/Desktop/DS340/logis")
  load("check.Rdata")
  load("raw_16.Rdata")
  f = function(x){ifelse(x == 1,"clickbait","good")}
  check[[1]][,4] = as.factor(sapply(check[[1]][,3],f))
  

  final = cbind(check[[1]],check[[2]],check[[3]])
  names(final) = c("SVM Target Title","SVM PostText","Logistic Regression All Text", "Neural Network", "Stacked Prediction","Actual")
  
  final$`Logistic Regression All Text` = sapply(final$`Logistic Regression All Text`,f)
  
  i = sample(1:dim(final)[1],1)
  output$try1 = renderTable(final[i,])
  output$raw = renderText(train22$postText[i])
  output$raw2 = renderText(train22$targetTitle[i])
  output$itt1 = renderText(as.character(i))
  
}
shinyApp(ui = ui, server = server) 