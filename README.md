# Amazon_Sentiment_Analysis_TextBlob_VADER_Topic_Modeling

## Abstract
Sentiment analysis is the process of extracting and classifying sentiments from textual data through the use of text analysis techniques. Sentiment analysis of Amazon reviews for Apple iPhones has a great deal of potential to provide insightful information that will help businesses and customers evaluate the performance to identify the customer’s likes and dislikes and to improve the product based on consumers’ feedback. In previous papers, authors used either rule or machine learning (ML) models for Amazon product reviews—where the lexicon-based approach has domain-specific limitations that limit its wide applicability in Amazon reviews, while learning-based approaches rely on annotated data. This project combines lexicon- and learning-based approaches to improve the results, where the output of sentiment analysis is given as the input of ML models. Rule-based sentiment analysis techniques, such as TextBlob and Valence Aware Dictionary and Sentiment Reasoner (VADER), are used for annotating the reviews, in which TextBlob (76.42%) outperforms VADER (66.68%), making it the target variable for ML. Subsequently, to identify prevalent themes, topic modeling is carried out using Latent Dirichlet Allocation and Latent Semantic Analysis, with the annotation of the prevalence of the top 30 words associated with topics, helping to understand the important words and concepts related to each topic. Moreover, 3 feature engineering methods, namely Bag of Words, Term Frequency-Inverse Document Frequency, and Word2Vec, have been used to extract useful features from reviews. Finally, ML models including the Logistic Regression, Decision Tree Classifier, Random Forest Classifier, K-Nearest Neighbor Classifier, and Multilayer Perceptron (MLP) are used to classify the positive and negative sentiments of the reviews. The highest accuracy of 97.97% is obtained by combining MLP with TextBlob and Bag of Word.

### Aims

This project aims to enhance the result of sentiment analysis by evaluating sentiment behind Amazon iPhone reviews, by combining rule, unsupervised, supervised, and deep learning approaches.
1.	This study assesses the efficacy of sentiment of iPhones on the Amazon website through a fusion of lexicon-ML technique, with the output of lexicon models (TextBlob and VADER) serving as target variables for ML (LR, DT, RF, KNN, and MLP) along with text normalization techniques (BOW, TF-IDF, and Word2Vec).
2.	Topic modeling using Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA) identifies the main topics and categorizes positive and negative reviews for the Amazon Apple iPhones.

### Objectives

The objectives of this research work are as follows:
1.	Retrieve the Amazon iPhone reviews dataset by using web scraping with the Beautiful Soup library.
2.	Categorize Amazon reviews into positive and negative using 2 rule-based sentiment analysis models, TextBlob and VADER, incorporating insights from a comprehensive literature review on sentiment analysis techniques given in a similar context.
3.	Evaluate the efficiency of 3 feature engineering approaches, comprising Bag of Words, TF-IDF, and Word2Vec, informed by a thorough review of the usage of these approaches in sentiment analysis and classification tasks.
4.	Utilizing machine learning models such as Logistic Regression for sentiment classification, applying insights from the relevant literature to implement these models in the context of sentiment analysis.
5.	Evaluate the performance matrices of the models involving accuracy, precision, recall, F1-score, and confusion matrix, considering benchmarks derived from the relative literature.
6.	Annotate the topics based on positive and negative reviews utilizing LDA, identifying topics by word distribution, and LSA, focusing on term co-occurrence patterns, including a literature exploration in the context of topic modeling.


## Methodology

In this study, the customer reviews on Apple iPhones are scraped by Beautiful Soup from the Amazon websites. Data preprocessing is performed including converting emoticons and lowercase; expanding contractions; deleting foreign language, null values, special characters, and stopwords; lemmatization; and tokenization. Sentiment scores are calculated by TextBlob and VADER. Word embedding techniques namely BOW, TF-IDF, and Word2Vec are employed. The output of TextBlob is given as a target variable and text normalization as input to 5 ML algorithms, including Logistic Regression (LR), Decision Tree (DT) Classifier, Random Forest (RF) Classifier, K-Nearest Neighbor (KNN), and Multilayer Perceptron (MLP). The workflow of the process is given in Figure 1.

Figure 1: Architecture of the procedures used in this project.

![image](https://github.com/user-attachments/assets/639c354c-3b7c-4ee8-afee-d5b6382542bd)

### Data collection

Webscraping is an automated technique that gathers data from the Amazon website without the need for mouse tappings, navigating, and scanning. Obtaining large amount of data can be difficult without automation. BeautifulSoup, a Python library, is used in this project, where the following operations take place: identify the url, inspect and choose the elements required, and store the data in CSV format. The distinguished architecture of each web pages and the issue of website changes are the challenges (Breuss, 2021).

#### Dataset

Amazon is one of the largest online e-commerce websites, where people can buy millions of products. Here, different versions of Apple iPhones are considered. The overview of the Amazon data along with data types and description is given in Table 1.

Table 1: The overview of collected data.

![image](https://github.com/user-attachments/assets/c15f91b3-129f-4a5e-b69b-8cd45cd32f9c)


### Data preprocessing

Data preprocessing is the procedure to convert raw data in order to improve the quality of data for further processing. In this study, it consists of 9 steps, which are as follows:

#### Convert emojis to words

Emojis are small pictorial representations of emotions, where users use those to convey the emotion. Hence, rather than removing, emoticons are converted to textual representation to retain the meaning, allowing non-verbal components to be expressed in written format (Table 2). Transforming emojis into words is a technique to improve communication, prevent miscommunication, and guarantee the intended message in formal communication.

Table 2: Example of emoticon conversion.

![image](https://github.com/user-attachments/assets/01ccbb8d-13bc-467f-b557-f8dbdb0cf677)


#### Convert text to lowercase

The data is standardized by converting all characters to lowercase in order to maintain consistency in output (Table 3), in which performing text mining is simplified. The terms such as ‘Apple’ and ‘apple’ would be treated differently, leading to redundancy. Converting to lowercase ensures that words are treated the same way, despite the original casing, which might be treated as separate features. Lowercasing helps in standardizing the text, ensuring that the focus is on the content rather than the formatting. It reduces the number of words and makes the model training more efficient. By not taking casing into account, it is easy to recognize patterns and relationships.

Table 3: Example for conversion of text to lowercase.

![image](https://github.com/user-attachments/assets/35f81646-90fd-4706-918f-bbf1579cbf51)









Breuss
