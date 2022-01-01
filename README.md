# Sentiment_analysis_of_tweets
Project to implement a Twitter sentiment analysis model for overcoming the challenges to identify the Twitter tweets text sentiments (positive, negative)

## Introduction
Sentiment analysis refers to identifying as well as classifying the sentiments that are expressed in the text source. Tweets are often useful in generating a vast amount of sentiment data upon analysis. These data are useful in understanding the opinion of the people about a variety of topics.
Therefore, we need to develop an Automated Machine Learning Sentiment Analysis Model in order to compute the customer perception using NER. Due to the presence of non-useful characters (collectively termed as the noise) along with useful data, it becomes difficult to implement models on them.

## Dataset
Twitter Sentiment Analysis
Detecting hatred tweets, provided by Analytics Vidhya
Link to the datasets :<br />
•	https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-atredspeech?select=train.csv

## Project Pipeline
•	Understand the problem statement<br />
•	Import Necessary Dependencies<br />
•	Read and Load the Dataset<br />
•	Exploratory Data Analysis<br />
•	Data Visualization of Target Variables<br />
•	Data Pre-processing<br />
•	Feature selection<br />
•	Splitting our data into Train and Test Subset<br />
•	Transforming Dataset using TF-IDF Vectorizer<br />
•	Model Building<br />
•	Determining which model is best (Hypothesis testing)

## Pre-processing
The pre-processing of the text data is an essential step as it makes the raw text ready for mining, i.e., it becomes easier to extract information from the text and apply machine learning algorithms to it.

## Cleaning data 
The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text.<br />
Further, we will be extracting numeric features from our data. This feature space is created using all the unique words present in the entire data. So, if we pre-process our data well, then we would be able to get a better-quality feature space.<br />
A) Removing Twitter Handles (@user)<br />
B) Removing Punctuations, Numbers, and Special Characters<br />
C) Removing Short Words<br />
D) Tokenization<br />
E) Stemming

## Exploratory Data Analysis & visualization 
Will explore the cleaned tweets text. Exploring and visualizing data, no matter whether its text or any other data, is an essential step in gaining insights.<br />
Did the following:<br />
•	Plot various graphs for positive and negative tweets on the basis of Label<br />
•	Visualizing the words in the tweets before and after cleaning through word clouds<br />
•	Plot ngrams for most occurring 1,2,3 words in our tweets<br />
•	Find the difference between the word frequency in our data through histograms<br />
•	Named entity recognition of tweets of different categories & Displaying sample observations<br />
•	Plotting named entities mentioned most times in Non-Offensive & Offensive tweets<br />
•	Finding count of text by each named entity of Non offensive & Offensive tweets<br />
•	Visualize most repetitive Non offensive & Offensive text phrases from each named entity<br />
•	Understanding the impact of Hashtags on tweets sentiment<br />
•	Displaying count of hashtags in each entity and plotting word counts of  Non offensive & Offensive hashtags<br />
•	Plotting Bar plots of top hashtag counts in Non-offensive & Offensive tweets<br />
•	Plotting  a pie chart counting number of rows containing a hashtag<br />

## Feature selection
TF-IDF Features<br />
TF-IDF works by penalizing the common words by assigning them lower weights while giving importance to words which are rare in the entire corpus but appear in good numbers in few documents.<br />
Important terms related to TF-IDF:<br />
•	TF = (Number of times term t appears in a document)/(Number of terms in the document)<br />
•	IDF = log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.<br />
•	TF-IDF = TF*IDF

## Model building:
In the problem statement we have used three different models respectively <br />
•	Logistic Regression<br />
•	Decision Tree Classifier<br />
•	Random Forest Classifier<br />
Want to try all the classifiers on the dataset and then try to find out the one which gives the best performance among them.

## Determining which model is best (Hypothesis testing)
After training the model we then apply the evaluation measures to check how the model is performing. Accordingly, we use the following evaluation parameters to check the performance of the models respectively :<br />
•	Accuracy Score<br />
•	F1 Scores<br />
•	Confusion Matrix <br />

### Hypothesis testing:
Examining machine learning models via statistical significance tests requires some expectations that will influence the statistical tests used.<br />
An approach to evaluate each model on the same k-fold cross-validation split of the data and calculates each split score. That would give a sample of ten scores for ten-fold cross-validation. Then, we can compare those scores using the paired statistical test.<br />
Due to using the same data rows to train the model more than once, the assumption of independence is violated; hence, the test would be biased.<br />
This statistical test could be adjusted to overcome the lack of independence. Also, the number of folds and repeats of the method can be configured to achieve a better sampling of model performance.<br />
Steps to hypothesis testing:<br />
The first step would be to to state the null hypothesis statement.<br />
H0: Both models have the same performance on the dataset.<br />
H1: Both models doesn’t have the same performance on the dataset.<br />
Significance level is 0.05<br />
let’s assume a significance threshold of α=0.05 for rejecting the null hypothesis that both algorithms perform equally well on the dataset and conduct the 5x2_cv _t_test.<br />
used the paired_ttest_5x2cv function from the evaluation module to calculate the t and p value for both models.

## Conclusion
Upon evaluating all the models we can conclude the following details i.e.
#### Accuracy: 
As far as the accuracy of the model is concerned Random Forest Classifier performs better than Decision Tree Classifier which in turn performs better than Logistic Regression.
#### F1-score: 
The F1 Scores for class 0 and class 1 are :<br />
(a) For class 0: Logistic Regression (accuracy = 0.97) < Decision Tree Classifier (accuracy =0.97) < Random Forest Classifier (accuracy = 0.98)<br />
(b) For class 1: Logistic Regression (accuracy = 0.94) < Decision Tree Classifier (accuracy =0.95) < Random Forest Classifier (accuracy = 0.96)<br />
We, therefore, conclude that the Random Forest Classifier is the best model for the above-given dataset.<br />


