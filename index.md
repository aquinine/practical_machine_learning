# Machine Learning
Heather Quinn  
August 11, 2016  

##Introduction

The Human Activity Recognition (HAR) project [1] is area of research that uses accelerometers to track movement in people.  It can be a challenge to detect what movement the test subjects are completing based on the accelerometer, as the data sets can be quite noisy.  Most HAR studies are trying to detect these common movements: sitting-down, standing-up, standing, walking, and sitting.  The Weight Lifting Exercise (WLE) data set [2], though, focuses on detecting whether the movement is executed correctly so that the quality of motion can be improved.

##Exploratory Data Analysis








```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```
There are two separate data sets in the WLE data set.  One is the training data and the other is the test data.  We broke the training data into a training set and a validation set, so that we can test the models on untrained data before using the model on the test data.  The training data used to train the model comprises 75% of the full training data, as this data set is very large there is still a reasonable amount of data for training the model.

It is necessary to clean the data before machine learning.  This particular data will have two issues with machine learning algorithms: missing values and number of columns.  Out of the 160 columns in the data set, there are 99 columns that are comprised of mostly missing values.  These missing values can cause machine learning algorithms to fail.  The other issue is that the random forest machine learning algorithms can only be executed with 53 or fewer categories.  After removing the 99 columns with missing values, there are still 61 columns, so more columns need to be removed.  We also chose to remove the columns with the primary key, user names, timestamps and windows.  The primary keys and windows are removed for being extraneous, and the user names are removed so that the machine learning output is not based on these users only.  We did attempt to keep the timestamps, because it could be useful to the machine learning model.  The models developed with the timestamps were 100% accurate, which concerned us that the model was over-fit.  We decided to remove those columns to avoid over-fitting.  It should be noted that these columns also have to be removed from the validation and test data to do the predictions.

## Machine Learning Model Selection 

We fit the training data with five models: random forest (rf), tree (rpart), quadratic discriminant analysis (qda), and gradient boosting (gbm).  We also attempted a linear discriminant model and and a combined model using random forest and gradient boosting (gam), but both had an accuracy of 40-50%.  We did both the randomForest and the caret version of the rf model, so that we could get the important variables from the randomForest and the cross-validation from the caret package.  The caret package version is used for the prediction.  The most important variables for the random forest in the plot below.  Of the top four variables three of them were related to the sensor on the belt.  


```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```


```r
varImpPlot(fit.rf2)
```

![](pml_project_files/figure-html/rf2-1.png)<!-- -->

Below we show the confusion matrices and out-of-sample accuracy for each of the four models using the validation data set.  While most of the models do a good job predicting A and B movements, only rf and gbm can predict C, D, and E movements with reasonable accuracy.  The rpart model is incapable of predicting C, D, and E movements.  For the rf model, the out-of-sample accuracy is 99%.  For the rpart model, the out-of-sample accuracy is 49%.  For the qda model, the out-of-sample accuracy is 89%.  For the gbm model, the out-of-sample accuracy is 96%.  Based on this analysis, the rf model is the best option for correctly predicting all five movements.


```r
fit.rf <- train(x,y, data=training,importance=TRUE, method="rf", trControl=fitControl)
pred.rf <- predict(fit.rf, validation)
confusionMatrix(pred.rf, validation$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    3    0    0    0
##          B    0  944    7    0    0
##          C    0    2  846   14    0
##          D    0    0    2  789    4
##          E    0    0    0    1  897
```

```r
confusionMatrix(pred.rf, validation$classe)$overall['Accuracy']
```

```
##  Accuracy 
## 0.9932708
```


```r
fit.gbm <- train(classe ~ ., data=training, method="gbm", verbose=FALSE, trControl=fitControl)
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```r
pred.gbm <- predict(fit.gbm, validation)
confusionMatrix(pred.gbm, validation$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1370   35    0    0    2
##          B   13  895   32    2    9
##          C    9   18  813   29    6
##          D    2    1    8  765   13
##          E    1    0    2    8  871
```

```r
confusionMatrix(pred.gbm, validation$classe)$overall['Accuracy']
```

```
##  Accuracy 
## 0.9612561
```


```r
fit.rpart <- train(classe~., method="rpart",data=training, trControl=fitControl)
```

```
## Loading required package: rpart
```

```r
pred.rpart <- predict(fit.rpart, validation)
confusionMatrix(pred.rpart, validation$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1246  383  378  340  121
##          B   32  328   43  161  120
##          C  112  238  434  303  249
##          D    0    0    0    0    0
##          E    5    0    0    0  411
```

```r
confusionMatrix(pred.rpart, validation$classe)$overall['Accuracy']
```

```
##  Accuracy 
## 0.4932708
```


```r
fit.qda <- train(classe~., method="qda",data=training, trControl=fitControl)
pred.qda <- predict(fit.qda, validation)
confusionMatrix(pred.qda, validation$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1266   46    1    3    0
##          B   60  798   53    4   27
##          C   38   97  791  102   41
##          D   29    1    2  692   20
##          E    2    7    8    3  813
```

```r
confusionMatrix(pred.qda, validation$classe)$overall['Accuracy']
```

```
##  Accuracy 
## 0.8890701
```

Finally, we look at the cross validation for all four models.  The models are cross validated using the "cv" method in the train control, which uses a 10-fold cross-validation technique.  The cross-validation data are shown below in the table and figure. For the rf model, the accuracy is 99.34% with a 95% confidence interval of (99.27%, 99.44%).  For the rpart model, the accuracy is 51% with a 95% confidence interval of (50%, 52%) .  For the qda model, the accuracy is 89.3% with a 95% confidence interval of (89.1%, 89.7%).  For the gbm model, the accuracy is 96.1% with a 95% confidence interval of (95.9%, 96.9%).  The cross-validated accuracy of the rf model is the best and the gbm model is second best. 


```r
resamps <- resamples(list(rf=fit.rf,
                          rpart=fit.rpart,
                          qda=fit.qda,
                          gbm=fit.gbm
                          ))
ss <- summary(resamps)
knitr::kable(ss[[3]]$Accuracy)
```

           Min.   1st Qu.   Median     Mean   3rd Qu.     Max.   NA's
------  -------  --------  -------  -------  --------  -------  -----
rf       0.9885    0.9927   0.9935   0.9934    0.9944   0.9959      0
rpart    0.4858    0.5035   0.5114   0.5114    0.5200   0.5303      0
qda      0.8838    0.8906   0.8933   0.8936    0.8971   0.9036      0
gbm      0.9545    0.9594   0.9616   0.9630    0.9693   0.9695      0



```r
trellis.par.set(caretTheme())
dotplot(resamps, metric = "Accuracy")
```

![](pml_project_files/figure-html/comp_plot-1.png)<!-- -->

##Prediction of Test Data

The prediction of the test data using the rf model are shown below:


```r
pred.rf.test <- predict(fit.rf, testing_smaller)
print(rbind(testing_smaller[1:20, 160], as.character(pred.rf.test)))
```

```
##      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13]
## [1,] "B"  "A"  "B"  "A"  "A"  "E"  "D"  "B"  "A"  "A"   "B"   "C"   "B"  
##      [,14] [,15] [,16] [,17] [,18] [,19] [,20]
## [1,] "A"   "E"   "E"   "A"   "B"   "B"   "B"
```

These results have 19/20 correct results for an accuracy of 95%.

#Conclusions

We applied five different models to the WLE training set, which was split into training and validation data sets.  We found that the random forest model could predict the validation data set with 99.34% accuracy with a 95% confidence interval of (99.27%, 99.44%) and an out-of-sample accuracy of 99% for the validation data set.  This accuracy was better than the rest of the models, so we used it to predict classe variable in the WLE test data set.  We were able to accurately predict the classe variable in 19/20 (95%) using this model.


##References

[1]  "Human Activity Recognition"  On web at http://groupware.les.inf.puc-rio.br/har, last accessed on 8/12/2016

[2]  Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
