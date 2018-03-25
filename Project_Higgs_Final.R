rm(list=ls())

library(class)
library('readxl')

Higgs_Subset23<-read_excel("/Users/SaiSanthosh/Desktop/CS513/Project_HiggsBoson/Higgs Boson/Higgs_Boson_Subset23.xlsx")
Higgs_Subset23[Higgs_Subset23== -999] <- NA
summary(Higgs_Subset23)
Higgs_Subset23<-na.omit(Higgs_Subset23)
Higgs_Subset23$Label<-as.factor(Higgs_Subset23$Label)
training<-Higgs_Subset23[,c(-1,-32)]
index<-sample(0.8*nrow(Higgs_Subset23))
Higgs_train<-training[index,]
Higgs_test<-training[-index,]

Test_Subset23<-read_excel("/Users/SaiSanthosh/Desktop/KDDM/Higgs Boson/Test_Subset23.xlsx")
Test_Subset23[Test_Subset23== -999] <- NA
Test_Subset23<-na.omit(Test_Subset23)
Test_Subset23<-Test_Subset23[,-1]
index<-sample(0.8*nrow(Test_Subset23))
test<-Test_Subset23[index,]
test$Label<-as.factor(test$Label)

library(randomForest)
n <- names(training)
f<-as.formula(paste("Label ~", paste(n[!n %in% "Label"], collapse = " + ")))


####caret package to find optimal solution for randomForest#######
control<- trainControl(method='repeatedcv', repeats=10, verboseIter=TRUE, classProbs = TRUE)
mod<- train(f, data=Higgs_Subset23, method= 'rf', trControl =control)


###RandomForest Model after finding optimal model using caret package############

mytree_2000<-randomForest(f,data=training,importance=TRUE, ntree=2000, mtry=16)
varImpPlot(mytree_2000)
plot(mytree_2000)

#######RandomForest with optimal model##########
mytree_1000<-randomForest(f,data=Higgs_train,importance=TRUE, ntree=1000, mtry=16)
varImpPlot(mytree_1000)
plot(mytree_1000)



prediction<-predict(mytree_1000,Higgs_test[,-31])
table(actual=test$Label,prediction)
wrong<-(test$Label!=prediction)
error_rate<-sum(wrong)/length(wrong)
error_rate
confusionMatrix(prediction, Higgs_test$Label)


names(getModelInfo())



## using Caret Package to find optimal knn method
install.packages('ISLR')
library(ISLR)

set.seed(400)
knncontrol <- trainControl(method="repeatedcv",repeats = 3, number=3, verboseIter = TRUE, classProbs = TRUE)
knnmod <- train(f, data = training , method = "knn", trControl = knncontrol, preProcess = c("center","scale"), tuneLength = 20)


#Output of kNN fit
knnmod
plot(knnmod)

length(Higgs_test[,31])

predict_knn<- knn(Higgs_train,Higgs_test[,-31],Higgs_test[,31],k=43)

predict_knn <- predict(knnmod, test[,-31])
confusionMatrix(predict_knn, test$Label)



######caret package for svm
library("caret")
svmcontrol <- trainControl(method="repeatedcv",repeats = 3, number=3, verboseIter = TRUE, classProbs = TRUE)
svmmodel <- train(f, method = "svmRadial", data = Higgs_train, trControl= svmcontrol)
predsvm <- predict(svmmodel, Higgs_test[,-31])
confusionMatrix(predsvm, Higgs_test$Label)

##################ANN

Higgs_Subset23_ANN<- Higgs_Subset23
#####setting 0-s and 1-b

Higgs_Subset23_ANN$Label <- setNames(0:1,c("s","b"))
library(neuralnet)
mynet<- neuralnet(f,data=Higgs_Subset23_ANN, hidden=16, linear.output = TRUE,stepmax = 1e09)

