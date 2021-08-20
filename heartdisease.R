# Performing PCA for Heart Disease dataset
library(readr)
# Importing the dataset
heartdisease <- read_csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\heart disease.csv')
View(heartdisease)
# Performing EDA for the heart disease dataset
# Checking for NA values
sum(is.na(heartdisease))
# There are no NA values in our dataset
# Plotting histogram to get skewness of the data
# We can plot only for continuous data
hist(heartdisease$trestbps,xlab = 'Trestbps',ylab = 'Frequency',main = 'Trestbps vs Frequency',col = 'red',border = 'black',breaks = 20)  # Histogram is normally skewed
hist(heartdisease$chol,xlab = 'Chol',ylab = 'Frequency',main = 'Chol vs Frequency',col = 'red',border = 'black',breaks = 20)  # Histogram is normally skewed)
hist(heartdisease$thalach,xlab = 'thalach',ylab = 'Frequency',main = 'thalach vs Frequency',col = 'red',border = 'black',breaks = 20)  # Histogram is negatively skewed
hist(heartdisease$oldpeak,xlab = 'oldpeak',ylab = 'Frequency',main = 'oldpeak vs Frequency',col = 'red',border = 'black',breaks = 20)  # Histogram is Positively skewed
# Plotting boxplot for identifying outliers
trestbps <- boxplot(heartdisease$trestbps,xlab='trestbps',ylab='Frequency',main='Trestbps vs Frequency',col = 'black',border = 'red')
trestbps$out  # Outliers are present in this column
# Therefore we replace the outliers by winsorization method
quant1 <- quantile(heartdisease$trestbps,probs = c(0.25,0.75))
quant1
wins1 <- quantile(heartdisease$trestbps,probs = c(0.05,0.95))
wins1
a1 <- 1.5*IQR(heartdisease$trestbps)
a1
b1 <- quant1[1] - a1
b1
c1 <- quant1[2] + a1
c1
# Replacing the outliers
heartdisease$trestbps[heartdisease$trestbps<b1] <- wins1[1]
heartdisease$trestbps[heartdisease$trestbps>c1] <- wins1[2]
d1 <- boxplot(heartdisease$trestbps)
d1$out  # Outliers are replaced
chol <- boxplot(heartdisease$chol,xlab='chol',ylab='Frequency',main='chol vs Frequency',col = 'black',border = 'red')
chol$out  # Outliers are present in this column
# therefore we replace the outliers by winsorization method
quant2 <- quantile(heartdisease$chol,probs = c(0.25,0.75))
quant2
wins2 <- quantile(heartdisease$chol,probs = c(0.05,0.95))
wins2
a2 <- 1.5*IQR(heartdisease$chol)
a2
b2 <- quant2[1] - a2
b2
c2 <- quant2[2] + a2
c2
# Replacing the outliers 
heartdisease$chol[heartdisease$chol<b2] <- wins2[1]
heartdisease$chol[heartdisease$chol>c2] <- wins2[2]
d2 <- boxplot(heartdisease$chol)
d2$out # Outliers are replaced
thalach <- boxplot(heartdisease$thalach,xlab='thalach',ylab='Frequency',main='thalach vs Frequency',col = 'black',border = 'red')
thalach$out  # There is a outlier in this column
# Replacing the outliers by winsorization method
quant3 <- quantile(heartdisease$thalach,probs = c(0.25,0.75))
quant3
wins3 <- quantile(heartdisease$thalach,probs = c(0.05,0.95))
wins3
a3 <- 1.5*IQR(heartdisease$thalach)
a3
b3 <- quant3[1] - a3
b3
c3 <- quant3[2] + a3
c3
# Replacing the outlier
heartdisease$thalach[heartdisease$thalach<b3] <- wins3[1]
heartdisease$thalach[heartdisease$thalach>c3] <- wins3[2]
d3 <- boxplot(heartdisease$thalach)
d3$out  # Outliers are replaced
oldpeak <- boxplot(heartdisease$oldpeak,xlab='oldpeak',ylab='Frequency',main='oldpeak vs Frequency',col = 'black',border = 'red')
oldpeak$out  # Outliers are present in this column
# Replacing the outliers by winsorization method
quant4 <- quantile(heartdisease$oldpeak,probs = c(0.25,0.75))
quant4
wins4 <- quantile(heartdisease$oldpeak,probs = c(0.05,0.95))
wins4
a4 <- 1.5*IQR(heartdisease$oldpeak)
a4
b4 <- quant4[1] - a4
b4
c4 <- quant4[2] + a4
c4
# Replacing the outliers
heartdisease$oldpeak[heartdisease$oldpeak<b4] <- wins4[1]
heartdisease$oldpeak[heartdisease$oldpeak>c4] <- wins4[2]
d4 <- boxplot(heartdisease$oldpeak)
d4$out  # Outliers are replaced
# Now we will check the normality of the data
qqnorm(heartdisease$trestbps)
qqline(heartdisease$trestbps)
# The data is normal.
qqnorm(heartdisease$chol)
qqline(heartdisease$chol)
# The data is normal
qqnorm(heartdisease$thalach)
qqline(heartdisease$thalach)
# The data is normal
qqnorm(heartdisease$oldpeak)
qqline(heartdisease$oldpeak)
# The data is non norma, therefore we apply is transformation
qqnorm(log(heartdisease$oldpeak),ylim = c(0,5))
qqline(log(heartdisease$oldpeak))
qqnorm(sqrt(heartdisease$oldpeak))
qqline(sqrt(heartdisease$oldpeak))
qqnorm((1/heartdisease$oldpeak),ylim = c(0,5))
qqline((1/heartdisease$oldpeak))
# Even after applying transformation the data is non normal
# Checking for variance in all columns
apply(heartdisease,2,var)
which(apply(heartdisease,2,var)==0)
# Therefore we have variance in all columns
# Now performing PCA on our dataset
pcaobj <- princomp(heartdisease,covmat = NULL,cor = TRUE,scores = TRUE)
str(pcaobj)
summary(pcaobj)
plot(pcaobj)    # This Plot will show the importance of principle components
biplot(pcaobj)
plot(cumsum(pcaobj$sdev*pcaobj$sdev)*100/(sum(pcaobj$sdev*pcaobj$sdev)),type = 'b')
pcaobj$scores
t3 <- pcaobj$scores[,1:3]
# Top 3 PCA scores
finalpca <- as.data.frame(t3)
View(finalpca)
colnames(finalpca)
# Scatter Diagram
plot(finalpca$Comp.1,finalpca$Comp.2)

# Performing clustering
# Distance matrix
d <- dist(finalpca,method = 'euclidean')
fit <- hclust(d,method = 'single')
# Plotting Dendrogram
plot(fit,hang = -1)
# Now creating clusters
groups <- cutree(fit,k=3)
rect.hclust(fit,k=3,border = 'red')
# Creating matrix for groups
hcluster <- as.matrix(groups)
# Adding membership column to final dataframe
membership <- data.frame(hcluster,heartdisease)
# Finding the aggregate
aggregate(heartdisease,by=list(membership$hcluster),FUN = mean)
# To get the output with having clustered group value column in it.
write_csv(membership, "hclustheartdisease.csv")
getwd() #to get working directory

# Performing K-Means
# Plotting Scree plot for deciding the k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss,kmeans(finalpca,centers = i)$tot.withinss)
}
twss
# Visualising elbow curve in scree plot
plot(2:8,twss,type = 'b',xlab = 'Number of Cluster',ylab = 'Within group of squares',main = 'K Means clustering scree plot')
# K means clustering solution
fit <- kmeans(finalpca,3)
str(fit)
fit$cluster
kmeansclust <- data.frame(fit$cluster,heartdisease)
aggregate(heartdisease,by=list(fit$cluster),FUN = mean)
# To get the output with having clustered group value column in it.
write_csv(membership, "kmeansheartdisease.csv")
getwd() #to get working directory

# From both Hierarichal Clustering and K-means clustering we can conclude that,
# The persons of higher age group have high cholesterol ratio and the Pharmaceutical company should focus on lower age group for implementing drugs
# because, the older the people age, higher the attack  ratio and very risky to implement on them.