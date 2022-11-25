# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: 
# NLP Classifier Subreddits 'OutOfTheLoop' & 'ExplainLikeI'mFive'


The objective of this project is to develop a predictive model to classify posts from two different subreddits. The choice of subreddits was made with care not to have too many overlapping terms while being careful not to be completely uncorrelated. The first chosen subreddit is called ‘Explain Like I'm Five’ [1] which people a request for a simple explanation to a complicated question or problems in general. Like they say: ‘layperson-friendly explanations. The second subreddit is ‘Out of The Loop’. This one, is A subreddit to help you keep up to date with what's going on with reddit and other stuff [2]. For this, two models were built using different supervised machine learning techniques such as Logistic Regression, Multinomial Naive Bayes in addition to hyper parameterization and tuning techniques such as Pipelines, GridSearch and Boosting in addition to using NPL for processing the text contained in each post subreddits


The metrics used to evaluate the success of the model will be the F1 score taking into account the baseline model score. The model will be developed in a training set and then validated in the test set, in which the F1 Score will be obtained.


When it comes to audience, we can think that this model can be useful for direct advertising to one or another specific user profile of each subreddit. A case study that could be thought of would be to take as a scenario that the user accessing the 'Explain me Like I'm Five' subreddit could potentially be someone who is learning the English language and has some difficulty with unusual words, in which case, can be a good choice for targeting English course ads. In this case, we would have as primary and secondary audience, owners of reddit sites and language course companies, respectively.

## About Data :.

The data used for the modeling and analysis of the project were obtained directly from the page of each of the respective subreddits. To achieve this data retention, we use the webscraping technique in combination with the reddit-specific API called Pushshift. In this process, 6,137 more records were obtained with 73 characteristics each, thus obtaining a final dataset of size (6,137, 73), that is, more than 6,000 posts were used to serve as a learning base for our model. This entire data collection process is documented in the Jupyter Notebook '00 Getting Data.ipynb' located in the code folder of this repository as well as the dataset itself in the data folder ('../data/subreddits_.csv').

* [`subreddit_cleaned_.csv`](./data/subreddit_cleaned_.csv) | [data dictionary](https://git.generalassemb.ly/sresende/project-3/blob/master/data/dict.txt)

Null and invalid values such as 'delete', 'Title' and 'Title.' have been removed from the dataset of the columns used in question. In a later step, it was noticed that links from several sites occurred with a certain frequency and given that we will be using NPL for the analysis of this data, and as these links do not add value or information to the model, we remove them and put them in a new column so that they can be processed in future works.  

In a later step, it was noticed that links from several sites occurred with a certain frequency and given that we will be using NPL for the analysis of this data, and as these links do not add value or information to the model, we remove them and put them in a new column so that they can be processed in future works. For language processing (NPL) we used several modules from the NLTK library such as word_tokenize, RegexpTokenizer, WordNetLemmatizer, and PorterStemmer. As the last step of data cleaning, we remove outliers and binarize our target.

 ## EDA and Conceptual Understanding :.

# ![](https://git.generalassemb.ly/sresende/project-3/blob/master/images/HistogramWordCount.png) ![](https://git.generalassemb.ly/sresende/project-3/blob/master/images/HistogramLength.png)

Continuing the processing, we vectorize all documents ('selftext' content posts) from dataset using TdifVectorizer(). This algorithm works in a different way than CountVectorizer(), putting different weights for the occurrence of each token considering overall document. It weights the word counts by a measure of how often they appear in the documents. To set this vectorizer, we set two main parameters as stop_words = ‘english’  and ngram_range = (1,3) . The first one, remove words without ‘meaning’ from our document and the second one look at the occurrence of one, two or three sequential words in the documents. 



# ![](https://git.generalassemb.ly/sresende/project-3/blob/master/images/ngram_1.png) ![](https://git.generalassemb.ly/sresende/project-3/blob/master/images/ngram_2.png)



####                    Most Importants Trigrams Occuring in Both Subreddits and It's Weights


# ![](https://git.generalassemb.ly/sresende/project-3/blob/master/images/TriGramsIlustration.png)

# ![](https://git.generalassemb.ly/sresende/project-3/blob/master/images/TrigramsSubreddits.png)



## Evaluation Models :.
As mentioned earlier, the metric chosen to evaluate the models was the F1 score. F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall.[3]

 F1 Score = 2 * (Recall * Precision) / (Recall + Precision)
 
### MULTINOMIAL NAÏVE BAEYS

Best Parameters: { \
'tvec__max_df': 0.9, 	# Maximum number of documents needed to include token (90%) \
‘tvec__max_features': 5000, # Maximum number of features fit. \
'tvec__min_df': 3, 		# Minimum number of documents needed to include token (3). \
'tvec__ngram_range': (1, 2), # Check (individual tokens) and also check 2-grams. \
'tvec__stop_words': None} 	  # Remove stop words (None) 

Train Score: 0.946  \
Test Score: 0.902  \
F1 Score: 0.9111 \
Accuracy: 0.902 

### LOGISTIC REGRESSION:

Best Parameters: \
{ \
'lr__penalty': 'l2',  \
'tvec__max_features': 4000,  \
'tvec__min_df': 4,  \
'tvec__ngram_range': (1, 2), \
'tvec__stop_words': None \
} 

Train Score: 0.9496  \
Test Score: 0.9 \
F1 Score: 0.9076 \
Accuracy: 0.9 

# ![](https://git.generalassemb.ly/sresende/project-3/blob/master/images/TableMetrics.png)

## Conclusion and Recommendations :.
Looking at the metrics in the training and test sets, we can verify that both models are not overfitting or have high bias and based on the F1 score metric (and also accuracy), we can say that both models perform very well in the forecast from these two subreddits. Another conclusion is that there doesn't seem to be much overlap in terms, otherwise it wouldn't perform as well.

## Future Works :.

•	Perform  a Test Validation: Get more data from each subreddit and placement a test  in unseen data; \
•	Add Information to Documents: Adding information to the model from links extracted from the documents (YouTube links); \
•	Others Subreddits: Applying to others subreddits to check if the model performs  well; \
•	Frame a Different Statement: Evaluate using different metrics to check the model performance (Sensitivity/Recall). Maybe we can think about a scenario where it makes sense to rank one class more than the other. For example, consider that the ELI5 profile is a person who is learning the English language and needs explanations about a certain subject in a simple and lay language. 

## Notes :.

This project is in continuos development and updating. If you want to check up for the most recents updates, try to look at diferents branches of it. Keep in mind that this fabulous project has reserved property rights and following creative commons guidelines.

____
References:

[1][*Explain Like I'm Five*](https://www.reddit.com/r/explainlikeimfive/)  \
[2][*Subreddit Out of the Loop*](https://www.reddit.com/r/OutOfTheLoop/)  \
[3][*F1 Score*](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/) 

