# Performing PCA on Wine dataset
library(readr)
wine <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\wine.csv')
View(wine)
# Performing EDA for the dataset
# Checking for NA values
sum(is.na(wine))
# There are no NA values in our dataset
# Plotting histogram for getting the skewness
hist(wine$Alcohol,xlab = 'Alcohol',ylab = 'Frequency',main = 'Alcohol vs Frequency', breaks = 20,col = 'blue',border = 'black')  # Histogram is normally skewed
hist(wine$Malic,xlab = 'Malic',ylab = 'Frequency',main = 'Malic vs Frequency',breaks = 20,col = 'blue',border = 'black')  # Histogram is positively skewed
hist(wine$Ash,xlab = 'Ash',ylab = 'Frequency',main = 'Ash vs Frequency',breaks = 20,col = 'blue',border = 'black')   # Histogram is normally skewed
hist(wine$Alcalinity,xlab = 'Alcalinity',ylab = 'Frequency',main = 'Alcalinity vs Frequency',breaks = 20,col = 'blue',border = 'black')  # Histogram is normally skewed
hist(wine$Magnesium,xlab = 'Magnesium',ylab = 'Frequency',main = 'Magnesium vs Frequency',breaks = 20,col = 'blue',border = 'black')   # Histogram is positively skewed
hist(wine$Phenols,xlab = 'Phenols',ylab = 'Frequency',main = 'Phenols vs Frequency',breaks = 20,col = 'blue',border = 'black')  # HIstogram is normally skewed
hist(wine$Flavanoids,xlab = 'Flavanoids',ylab = 'Frequency',main = 'Flavanoids vs Frequency',breaks = 20,col = 'blue',border = 'black')  # Histogram is normally skewed
hist(wine$Nonflavanoids,xlab = 'Non Flavanoids',ylab = 'Frequency',main = 'Non Flavanoids vs Frequency',breaks = 20,col = 'blue',border = 'black')   # Histogram is positively skewed
hist(wine$Proanthocyanins,xlab = 'Proanthocyanins',ylab = 'Frequency',main = 'Proanthocyanins vs Frequency',breaks = 20,col = 'blue',border = 'black')   # Histogram is positively skewed
hist(wine$Color,xlab = 'Color',ylab = 'Frequency',main = 'Color vs Frequency',breaks = 20,col = 'blue',border = 'black')   # Histogram is positively skewed
hist(wine$Hue,xlab = 'Hue',ylab = 'Frequency',main = 'Hue vs Frequency',breaks = 20,col = 'blue',border = 'black')   # Histogram is negatively skewed
hist(wine$Dilution,xlab = 'Dilution',ylab = 'Frequency',main = 'Dilution vs Frequency',breaks = 20,col = 'blue',border = 'black')   # Histogram is negatively skewed
hist(wine$Proline,xlab = 'Proline',ylab = 'Frequency',main = 'Proline vs Frequency',breaks = 20,col = 'blue',border = 'black')   # Histogram is positively skewed
# Plotting boxplot for finding the outliers
alcohol <- boxplot(wine$Alcohol,xlab='Alcohol',ylab='Frequency',main='Alcohol vs Frequency',col = 'black',border = 'blue')
alcohol$out
# There are no outliers in Alcohol column
Malic <- boxplot(wine$Malic,xlab='Malic',ylab='Frequency',main='Malic vs Frequency',col = 'black',border = 'blue')
Malic$out  # There are outliers in Malic column
# We will remove the outliers by winsorization method
quant1 <- quantile(wine$Malic,probs = c(0.25,0.75))
quant1
wins1 <- quantile(wine$Malic,probs = c(0.05,0.95))
wins1
a1 <- 1.5*IQR(wine$Malic)
a1
b1 <- quant1[1] - a1
b1
c1 <- quant1[2] + a1
c1
# Replacing the outliers
wine$Malic[wine$Malic<b1] <- wins1[1]
wine$Malic[wine$Malic>c1] <- wins1[2]
d1 <- boxplot(wine$Malic)
d1$out
# Outliers have been replaced
ash <- boxplot(wine$Ash,xlab='Ash',ylab='Frequency',main='Ash vs Frequency',col = 'black',border = 'blue')
ash$out
# There are outliers in Ash column
# Therefore we will remove the outliers by winsorization
quant2 <- quantile(wine$Ash,probs = c(0.25,0.75))
quant2
wins2 <- quantile(wine$Ash,probs = c(0.05,0.95))
wins2
a2 <- 1.5*IQR(wine$Ash)
a2
b2 <- quant2[1] - a2
b2
c2 <- quant2[2] + a2
c2
# Replacing the outliers
wine$Ash[wine$Ash<b2] <- wins2[1]
wine$Ash[wine$Ash>c2] <- wins2[2]
d2 <- boxplot(wine$Ash)
d2$out
# Outliers are replaced
alcalinity <- boxplot(wine$Alcalinity,xlab='Alcalinity',ylab='Frequency',main='Alcalinity vs Frequency',col = 'black',border = 'blue')
alcalinity$out
# There are outliers in Alcalinity column
# Therefore we will replacing by winsorization 
quant3 <- quantile(wine$Alcalinity,probs = c(0.25,0.75))
quant3
wins3 <- quantile(wine$Alcalinity,probs = c(0.05,0.95))
wins3
a3 <- 1.5*IQR(wine$Alcalinity)
a3
b3 <- quant3[1] - a3
b3
c3 <- quant3[2] + a3
c3
# REplacing the outliers
wine$Alcalinity[wine$Alcalinity<b3] <- wins3[1]
wine$Alcalinity[wine$Alcalinity>c3] <- wins3[2]
d3 <- boxplot(wine$Alcalinity)
d3$out
# Therefore outliers are replaced
mag <- boxplot(wine$Magnesium,xlab='Magnesium',ylab='Frequency',main='Magnesium vs Frequency',col = 'black',border = 'blue')
mag$out
# Outliers are there in Magnesium column
# Replacing them by winsorization
quant4 <- quantile(wine$Magnesium,probs = c(0.25,0.75))
quant4
wins4 <- quantile(wine$Magnesium,probs = c(0.05,0.95))
wins4
a4 <- 1.5*IQR(wine$Magnesium)
a4
b4 <- quant4[1] - a4
b4
c4 <- quant4[2] + a4
c4
# Replacing the outliers
wine$Magnesium[wine$Magnesium<b4] <- wins4[1]
wine$Magnesium[wine$Magnesium>c4] <- wins4[2]
d4 <- boxplot(wine$Magnesium)
d4$out
# Outliers are replaced
phenols <- boxplot(wine$Phenols,xlab='phenols',ylab='Frequency',main='Phenols vs Frequency',col = 'black',border = 'blue')
phenols$out
# There are no outliers in Phenols column
flavanoids <- boxplot(wine$Flavanoids,xlab='Flavanoids',ylab='Frequency',main='Flavanoids vs Frequency',col = 'black',border = 'blue')
flavanoids$out
# There are no outliers in Flavanoids coloumn
nonflava <- boxplot(wine$Nonflavanoids,xlab='Non Flavanoids',ylab='Frequency',main='Non Flavanoids vs Frequency',col = 'black',border = 'blue')
nonflava$out 
# There are no outliers in Non Flavanoids column
proantho <- boxplot(wine$Proanthocyanins,xlab='proanthocyanins',ylab='Frequency',main='Proanthocyanins vs Frequency',col = 'black',border = 'blue')
proantho$out
# There are outliers in Proanthocyanins column
# Therefore we will do winsorization
quant5 <- quantile(wine$Proanthocyanins,probs = c(0.25,0.75))
quant5
wins5 <- quantile(wine$Proanthocyanins,probs = c(0.05,0.95))
wins5
a5 <- 1.5*IQR(wine$Proanthocyanins)
a5
b5 <- quant5[1] - a5
b5
c5 <- quant5[2] + a5
c5
# Replacing the outliers
wine$Proanthocyanins[wine$Proanthocyanins<b5] <- wins5[1]
wine$Proanthocyanins[wine$Proanthocyanins>c5] <- wins5[2]
d5 <- boxplot(wine$Proanthocyanins)
d5$out
# Outliers are replaced
color <- boxplot(wine$Color,xlab='Color',ylab='Frequency',main='Color vs Frequency',col = 'black',border = 'blue')
color$out
# There are outliers in Color column
# Therefore we will replace by winsorization method
quant6 <- quantile(wine$Color,probs = c(0.25,0.75))
quant6
wins6 <- quantile(wine$Color,probs = c(0.05,0.95))
wins6
a6 <- 1.5*IQR(wine$Color)
a6
b6 <- quant6[1] - a6
b6
c6 <- quant6[2] + a6
c6
# Replacing the outliers
wine$Color[wine$Color<b6] <- wins6[1]
wine$Color[wine$Color>c6] <- wins6[2]
d6 <- boxplot(wine$Color)
d6$out
# Outliers are replaced
hue <- boxplot(wine$Hue,xlab='Hue',ylab='Frequency',main= 'hue vs Frequency',col = 'black',border = 'blue')
hue$out
# There is a outlier in Hue column
quant7 <- quantile(wine$Hue,probs = c(0.25,0.75))
quant7
wins7 <- quantile(wine$Hue,probs = c(0.05,0.95))
wins7
a7 <- 1.5*IQR(wine$Hue)
a7
b7 <- quant7[1] - a7
b7
c7 <- quant7[2] + a7
c7
# Replacing the outliers
wine$Hue[wine$Hue<b7] <- wins7[1]
wine$Hue[wine$Hue>c7] <- wins7[2]
d7 <- boxplot(wine$Hue)
d7$out
# Outliers are replaced
dil <- boxplot(wine$Dilution,xlab='Dilution',ylab='Frequency',main='Dilution vs Frequency',col = 'black',border = 'blue')
dil$out
# There are no outliers in Dilution column
proline <- boxplot(wine$Proline,xlab='Proline',ylab='Frequency',main='Proline vs Frequency',col = 'black',border = 'blue')
proline$out
# There are no outliers in Proline column
# Checking the Normality of the data
qqnorm(wine$Alcohol)
qqline(wine$Alcohol)
# Alcohol data is Normal
qqnorm(wine$Malic)
qqline(wine$Malic)
# Malic data is non normal therefore we apply transformation 
qqnorm(log(wine$Malic))
qqline(log(wine$Malic))
qqnorm(sqrt(wine$Malic))
qqline(sqrt(wine$Malic))
qqnorm((1/wine$Malic))
qqline((1/wine$Malic))
# Even after applying the transformation the data is non normal
qqnorm(wine$Ash)
qqline(wine$Ash)
# Ash data is normal
qqnorm(wine$Alcalinity)
qqline(wine$Alcalinity)
# The data is normal
qqnorm(wine$Magnesium)
qqline(wine$Magnesium)
# The data is normal
qqnorm(wine$Phenols)
qqline(wine$Phenols)
# The data is normal
qqnorm(wine$Flavanoids)
qqline(wine$Flavanoids)
# The data is normal
qqnorm(wine$Nonflavanoids)
qqline(wine$Nonflavanoids)
# The data is normal 
qqnorm(wine$Proanthocyanins)
qqline(wine$Proanthocyanins)
# The data is normal
qqnorm(wine$Color)
qqline(wine$Color)
# The data is normal
qqnorm(wine$Hue)
qqline(wine$Hue)
# The data is normal
qqnorm(wine$Dilution)
qqline(wine$Dilution)
# The data is non normal
qqnorm(log(wine$Dilution))
qqline(log(wine$Dilution))
qqnorm(sqrt(wine$Dilution))
qqline(sqrt(wine$Dilution))
qqnorm((1/wine$Dilution))
qqline((1/wine$Dilution))
# Even after applying the transformations the data is non normal
qqnorm(wine$Proline)
qqline(wine$Proline)
# The data is non normal therefore we apply transformation
qqnorm(log(wine$Proline))
qqline(log(wine$Proline))
qqnorm(sqrt(wine$Proline))
qqline(sqrt(wine$Proline))
qqnorm((1/wine$Proline))
qqline((1/wine$Proline))
# After applying reciprocal the data is normal
wine$Proline <- (1/wine$Proline)
# Checking for variance in all columns
apply(wine, 2,var)
which(apply(wine,2,var)==0)
# We have variance in all columns
# We will drop first column
wine1 <- wine[-1]
# Finding Principle Components
pcaobj <- princomp(wine1,cor = TRUE,scores = TRUE,covmat = NULL)
str(pcaobj)
summary(pcaobj)
loadings(pcaobj)
plot(pcaobj)    # This Plot will show the importance of principle components
biplot(pcaobj)
plot(cumsum(pcaobj$sdev*pcaobj$sdev)*100/(sum(pcaobj$sdev*pcaobj$sdev)),type = 'b')
pcaobj$scores
s <- pcaobj$scores[,1:3]
# Top 3 PCA scores
pcafinal <- as.data.frame(cbind(wine[,1],s))
View(pcafinal)
# Plotting scatter diagram
plot(pcafinal$Comp.1,pcafinal$Comp.2)

# Performing Clustering
# Distance matrix
d <- dist(pcafinal,method = 'euclidean')
fit <- hclust(d,method = 'complete')
# Plotting Dendrogram
plot(fit,hang = -1)
groups <- cutree(fit,k=3)  # Making 3 groups
rect.hclust(fit,k=3,border = 'red')
membership <- as.matrix(groups)
finalhclust <- data.frame(membership,wine1)
aggregate(wine1[,1:13],by=list(finalhclust$membership),FUN = mean)
# To get the output with having clustered group value column in it.
write_csv(final, "hclustwine.csv")
getwd() #to get working directory

# Performing K-Means 
# Plotting elbow curve for deciding k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss,kmeans(pcafinal,centers = i)$tot.withinss)
}
twss
plot(2:8,twss,type = 'b',xlab = 'Number of clusters',ylab = 'within sum of squares',main = 'K Means clustering scree plot')
# Clustering Solution
fit <- kmeans(pcafinal,3)
str(fit)
fit$cluster
kmeansfinal <- data.frame(fit$cluster,wine)
aggregate(wine1[,1:13],by = list(fit$cluster),FUN = mean)
# To get the output with having clustered group value column in it.
write_csv(final, "hclustwine.csv")
getwd() #to get working directory

# By both Hierarchial and K-Means clustering we can conclude that,
# The wine belonging to group 1 has high alcohol content, low Alcalinity and color, 
# therefore we may consider this group as Premium Quality Wines

# The wine belonging to group 2 have moderate alcohol content,ash content and color,
# therefore this group maybe considered as White Wine Section

# The wine belonging to group 3 are less alcoholic and high color and ash,
# therefore this section can be categorised as Low Quality Wines.


