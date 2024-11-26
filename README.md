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

#### Expand contractions

The contractions are expanded to retain the meaning, particularly negative emotions (Table 4). Expanding contractions help during tokenization because each word is treated as a separate token where equal weightage is given for negative terms, especially ‘not’. Understanding the full form of words enhances the precision of the text’s analysis, which is crucial in sentiment analysis.

Table 4: Example for before and after expanding contractions.

![image](https://github.com/user-attachments/assets/f7489bc2-f652-497a-be00-ef1bbcf26421)


#### Delete foreign language

In this method, only English-language reviews are retained to remain consistent throughout (Table 5). NLP models are trained on specific language, so having it may confuse the performance of the models. A foreign language can add noise (errors or inaccuracies) to the data, enabling it more challenging to extract meaningful insights or patterns. Analyzing foreign text is expensive to compute. 

Table 5: Example for removing foreign language.

![image](https://github.com/user-attachments/assets/c7215f7f-ed37-4c3d-abce-381ef6024a94)


#### Handling null values

As these reviews are textual data, the missing information is removed as no other technique, such as imputation, can be performed (Table 6). While performing the models with null values, the efficiency of the models leads to errors.

Table 6: Example for null values removal.

![image](https://github.com/user-attachments/assets/d6dfb0b5-9571-4a38-9928-a4bb8696c562)


#### Remove special characters

Removing non-alphabetical characters, symbols, and whitespaces (e.g., comma, plus, colon, semicolon, apostrophe, hash, ampersand, quotes, and so on) is an important task as it may not contribute any meaning to the text (Table 7). With punctuation, the same words are treated as different entities. By removing it, the effectiveness of the memory can be increased.

Table 7: Remove punctuations.

![image](https://github.com/user-attachments/assets/95e81685-fa81-472c-9a72-d2fbe20cbe57)


#### Stopword removal

Before customization of stopwords: Stopwords are commonly used terms, such as articles and prepositions, that do not give meaning to the text (Table 8). Removing stopwords can reduce the size of the document, while still retaining the important words and meaning of the terms.

Table 8: Example of how stopwords are before and after removal.

![image](https://github.com/user-attachments/assets/d339e1cd-8a7c-4ac1-8068-9beee87903d0)

After removing stopwords, the top 20 most commonly used terms are checked and plotted (Figure 2) to check the weightage of the top 20 words.

Figure 2: Top 20 most frequently used terms.

![image](https://github.com/user-attachments/assets/2ef5bab1-2466-4dda-bc3e-9b7c802185b0)


Customizing stopwords: As the dataset solely focuses on the Apple iPhone, it is not necessary to have terms like phone, iPhone, apple, and amazon as these are product names used in this work and did not contribute to sentiment analysis. Hence, the words are removed, and the top 20 is plotted (Figure 3).

Figure 3: Top 20 most commonly used terms after customizing stopwords.

![image](https://github.com/user-attachments/assets/6791c416-841f-406b-9fbe-7d11c50cd769)


#### Lemmatization

Lemmatization reduces the words to their base form, particularly for verbs and adverbs. Stemming is not used here since it has a limitation of over- or under-stemming (Table 9).

Table 9: Example for before and after lemmatization.

![image](https://github.com/user-attachments/assets/291fbc22-478a-4f78-9714-c2eb041f0e12)


#### Tokenization

Tokenization is used to tokenize each word separately in order to derive meaning from each term (Table 10).

Table 10: Example for before and after tokenization.

![image](https://github.com/user-attachments/assets/dc251718-6f63-430a-937d-6eb44a521d4c)


### Exploratory data analysis

To get the hidden trends, patterns, and relationships, exploratory data analysis is utilized where the data can be visualized to gain insights and understand the structure of data.


#### Word Cloud

Word Cloud displays the frequently occurring words in a visual format. As shown in Figure 4, larger words, font size, and size variation in the cloud indicate more important or frequently used words, e.g., battery life and brand new. Colors are used to convey different attributes; here, different colors indicate different frequency terms. Words that are close to one another may have associations that point to existing relationships in the text.

Figure 4: Word Cloud based on more number of occurrences.

![image](https://github.com/user-attachments/assets/fdcfe867-6f69-4b58-a382-773c68d90f9a)


#### Frequency of ratings

A bar chart is plotted to get the frequency of ratings (categorical variables). Here, rating 5 has the highest frequency distribution of above 10,000 (Figure 5) and rating 2 has the lowest frequency distribution of approximately 1,500, which indicates that most of the reviews are positive.

Figure 5: Bar plot for rating frequency.

![image](https://github.com/user-attachments/assets/b8065e5b-53e3-4f55-9330-1c5373184b89)


#### Central tendency and spread of data

Boxplot is used to check for central tendency and spread of the data, i.e., to check where most of the data is lying. In this, the majority of ratings range between 3 and 5, indicating more positive sentiment (Figure 6).

Figure 6: Box plot to check central tendency and spread.

![image](https://github.com/user-attachments/assets/2ab1b684-39ae-40ab-91b6-1072d875525d)


#### Frequency of products

A bar graph is also used to determine the frequency of categorical values (product), where Apple iPhone 8 has the highest number of reviews (Figure 7) and Apple iPhone X has the least number of reviews.

Figure 7: Bar graph for frequency distribution of reviewed products.

![image](https://github.com/user-attachments/assets/c18409a5-7410-423f-92a2-ba8102027099)


#### Average ratings

The average rating of all products with a number of reviews is displayed in Figure 8, where the majority of products have either 3.5 or 4.5 average ratings, meaning that the range varies from 3 to 5. The colors indicate the number of reviews.

Figure 8: The average rating of all products with reviews.

![image](https://github.com/user-attachments/assets/593edd6a-1919-4779-8d28-ce256baf8313)


#### Ratings distribution

Bar plots for all products against each rating for all products are plotted in Figure 9 to get an idea on how each product has been rated, in which iPhone 13 Pro has more number of 1-star ratings, on the other hand, iPhone 8 has the highest 5-star ratings.

Figure 9: Distribution of 1- to 5-star ratings.

![image](https://github.com/user-attachments/assets/cb79de77-6db4-41c5-8bb1-d95cdf36b021)


### Sentiment analysis

Sentiment analysis, also known as opinion mining, is the process of identifying and classifying the sentiment as positive, neutral, or negative in a text, where the aim is to map words to emotions, which is already pre-built. It is used to analyze the opinions, emotions, and feelings communicated in text data, such as textual data from social media platforms and customers’ feedback on an e-commerce website. In this work, TextBlob and VADER are used to perform sentiment analysis.

#### TextBlob

TextBlob is one of the lexicon-based methods, where the sentiment is determined by both the strength of individual words and how words provide a certain feeling, by computing sentiment score to analyze positive, neutral, and negative texts. TextBlob provides labels to help in detailed analysis and calculates the polarity and subjectivity scores of the text.

Polarity exists between the range of -1 to 1, where negativity is represented by -1 and positivity is denoted as 1. When negative terms like “not” are used, TextBlob inverts the polarity from positive to negative. The subjectivity score ranges between 0 and 1, where 0 indicates a highly objective phrase (i.e., observable facts) and 1 indicates a highly subjective phrase (i.e., own opinions) (Shah, 2020).


#### VADER

Valence Aware Dictionary and Sentiment Reasoner (VADER) is another rule-based approach. In VADER, the compound score is obtained by adding the individual word scores ranging from -4 (majorly negative) to 4 (mostly positive), where -4 is majorly negative sentiment and 4 indicates mostly positive sentiment. Then, it is normalized to the range of -1 to 1, where negativity is implied by -1, neutrality by 0, and positivity by 1 (Calderon, 2017).


### Topic Modeling

Topic modeling is used to find the main subjects by identifying patterns and trends, involving the frequency of words and grouping similar words. By using this technique, it is easy to understand what is discussed in each topic. Topic modeling is divided into 2 parts: (i) creating a list of topics to provide a high-level overview of the topic discussed, and (ii) grouping a set of documents based on topics (Pascual, 2019).


#### Latent Dirichlet Allocation (LDA)

Latent Dirichlet Allocation (LDA) is built based on the distributional hypothesis, where the same topics use the same terms, and the statistical mixture hypothesis, indicating that documents contain various topics. LDA helps identify what each document is about by connecting it to a group of topics including many words in the document. However, LDA does not consider the arrangement of words and the structure of documents; instead, documents are treated as a collection of individual words.

#### Latent Semantic Analysis (LSA)

Latent Semantic Analysis (LSA) is similar to LDA in following distributional hypothesis, when terms appear together, it indicates that those terms have the same meaning throughout the document. LSA calculates how often terms appear in both a single document and collection, expecting that documents with similar content will display similar frequencies of words. Similar to LDA, LSA has the limitation of finding how words are arranged or the different meanings those terms might have; rather, treating each document as a bunch of separate words.


### Feature engineering

Word embedding, a widely used method for representing text-based lexicon, is used to transform words into numerals, and these vectors can be utilized by ML methods for processing. Word embeddings collect the meaning-related aspects of words, which makes machines to easily understand the complexity and meaning of individual words. For example, it uses identical numerical arrays to represent words that frequently occur together in the same document in order to record relationships with words (Shivanandhan, 2023).


#### Bag of Words

Text is unstructured when retrieved from a website, but ML prefers structured text; hence, BOW transforms variable-length texts into fixed-length numbers using CountVectorizer and it focuses on the frequencies of words rather than their specific location and English grammar rules (Figure 10). It consists of 2 components: a list of well-known terms, and an evaluation of the occurrences of those well-known terms. The limitations include creating a scoring system for the existence of these known words and designing the list of recognized tokens (Brownlee, 2019).

Figure 10: Illustration for Bag of Words. From (Somarouthu, 2023).

![image](https://github.com/user-attachments/assets/2d799c8f-ebd6-495c-95e5-803dddf4ead9)

In BOW, the data cleaning processes take place involve the elimination of stopwords, emoticons, numerals, punctuation marks, and white spaces and the conversion of text to lowercase. Then, a number of occurrences of each value are calculated. Finally, vectorization makes a list where each word represents a different word in a group of documents and the number of each word displays the number of times the word appears in the text.

The limitations of BOW are as follows: understanding content and relationships is challenging since words that have identical meanings can be interpreted as having entirely different meanings; BOW does not consider the meaning and sense of words. In a collection of text, there are many different words, creating lengthy but sparser arrays, thereby making the system work less efficiently (Somarouthu, 2023).


#### TF-IDF

Term Frequency-Inverse Document Frequency (TF-IDF), a widely used statistical analysis for the extraction of information, evaluates a term’s weightage in a document with regard to a collection of documents. The text vectorization method assigns numerical values representing weightage to terms within a document. TF is the ratio of the number of times the term appears in the document to the total number of words in that document. IDF is the ratio of the logarithmic value ratio of the number of documents in the collection of documents to the number of documents in the group with the word. The product of TF and IDF forms the TF-IDF (Karabiber, n.d.).

The following steps are considered to process TF-IDF: Data cleaning includes tokenizing, deleting punctuation and special characters, transforming to lowercase, and removing stopwords. Enhancing the scores of TF-IDF by customizing the stopword list to remove terms related to task and industry ensures that important and appropriate terms are considered. By standardizing the scores to be on the same scale, scaling scores in the interval is helpful for greater understanding and evaluation. Fine-tuning parameters include adjusting components namely n-gram handling or weighting schemes; investigating these changes leads to better results customized for particular tasks (Khetan, 2023).

Selecting words with the highest score in TF-IDF allows important terms to be extracted from the document, providing an outline of important topics and obtaining information. Examining scores of TF-IDF for user choices and description of items allows to suggest relevant words by evaluating their similarity. The disadvantages of TF-IDF are the same as BOW.


#### Word2Vec

Word2Vec detects the meaning of the terms, unlike BOW and TF-IDF. This is done by mapping words into dense vectors in a continuous vector space, and terms close to one another are considered the same. Word2Vec comprises 2 models: continuous BOW (CBOW) and Skip-gram. CBOW uses the surrounding words to forecast the target word, while Skip-gram, in contrast, operates by predicting the context terms with the target word. Both undergo the following steps: 1. Preprocess the text by tokenization of words, deletion of special characters and punctuations, and transformation of text to lowercase. 2. Create input and output pairs by sliding a fixed window size over the data to predict output data in CBOW, whereas the same process is done but for input data with surrounding words in Skip-gram. 3. Convert input and output terms to one-hot encoded vectors. 4. Construct a neural network with an input layer, which converts one-hot encoded text to dense vectors; hidden layers, which conduct transformations non-linearly; and an output layer, which predicts the target term. 5. Use methods like stochastic gradient descent or Adam optimizer to minimize the difference between estimated and observed target words. 6. Enhance contextual association in the vector space, where the weights of the input layer after training illustrate word embedding (Dube, n.d.).

Word2Vec facilitates the degree of similarity by calculating cosine similarity, which is a similarity between two vectors ranging from completely dissimilar (-1) to completely similar (1), and relationships between words, such as dogs to cats. The usage includes recommendation engines, data retrieval, and query-solving applications. In text categorization and sentiment analysis, Word2Vec embedding shows an improvement in meaningful communication, representing words as continuous vectors and encoding information on meanings. It facilitates linguistic and automated translation by arranging words in foreign languages, allowing to identify the similar words in foreign languages, thereby improving accuracy and context. It collects the resemblances in meaning and interconnections within the context of text, in order to improve the accuracy of recognizing and categorizing named entities in text, namely the name of the person, place, and institutions. It is helpful in clustering related data or extracting meaningful topics to create document-level embedding and to enable efficient methods for text grouping and topic analysis, i.e., analyzing text data, suggesting relevant content, and understanding collections of documents (Gomede, n.d.).



### ML models

ML models are employed to learn complex language patterns, adapt to diverse context, and continuously improve with information. Unlike previous studies, where most of the articles did not use neural networks, in this research, 5 ML models are taken into account, where 4 are supervised (i.e., interpretability) models (LR, DT, RF, and KNN) and 1 is a neural network for complexity (MLP).


#### Logistic Regression

Logistic Regression (LR) is a classification method utilized in supervised ML techniques, in which models are trained using a labeled dataset that has an answer included so the accuracy can be determined. Here, LR is used to solve the binomial prediction, even though there are other types, namely multinomial and ordinal. The outcome of the prediction models has only two values, in this case, either positive or negative, indicating values 1 or 0, respectively. In LR, the sigmoid function is utilized to convert predicted values into probabilities within the range of 0 to 1, showing a positive derivative (Raj, 2020).

When dealing with a large number of categorical values, LR becomes difficult and susceptible to overfitting. Its application is restricted to linear problems, and it performs badly when independent variables show correlation but no correlation with the target value.


#### Decision Tree Classifier

A Decision Tree (DT) is a diagram in hierarchical order resembling a flowchart, in which a root node signifies a feature, a branch represents a decision rule, and an individual leaf node corresponds to the final results. The use of Boolean logic and graphical representation enhances understanding. Its ability to work with different data types, such as discrete or continuous values, and its ability to apply thresholds to convert continuous into categorical values and handle null values all demonstrate its adaptability. It is flexible to use both regression and classification tasks. It does not take into account the relationship between features, even though there is a correlation, DT will use only one feature for splitting.

The problem of overfitting and poor generalization can be addressed by pre- and post-pruning to remove subtrees with poor data and to stop the growth of the tree when the data is not sufficient. DT with slightly different data may differ, yet this approach is limited by its tendency to generate highly correlated data. When the decisions are made based on immediate choices but without considering long-term impact, training them will incur higher costs (IBM, n.d.).


#### Random Forest Classifier

A Random Forest (RF) Classifier constructs many decision trees and combines their output to achieve a prediction that is both more precise and reliable. By choosing the best feature and from a random subset of features rather than looking for an important feature when splitting a node, it adds additional randomness to the tree growth process by including randomly determined thresholds for individual features, encouraging diversity and producing a better model. It performs well in both regression and classification tasks and offers clear insights into the relative weights assigned to input features due to adaptability. Understanding the hyperparameters is not complex. With a sufficient number of trees, RF avoids overfitting of the model. One major disadvantage is that the RF may become unresponsive or ineffective for real-time predictions when an extensive number of trees are used (Donges, 2023).


#### K-Nearest Neighbor Classifier

The K-Nearest Neighbor (KNN) uses a single data point’s proximity to classify or predict how it will be grouped. In order to avoid overfitting or underfitting, it is important to balance the selection of k values. Choosing a lower k value may result in high variance but low bias, whereas a larger k value may produce both high variance and lower bias. Due to its ease and accuracy, the algorithm is easy to implement. It adapts to new training samples by incorporating them into memory, ensuring that adjustments are made to data. KNN needs to specify a k value and choose a distance metric, so it does not require many hyperparameters. The limitation of KNN is that it uses a larger amount of memory and storage of data (IBM, n.d.).


#### Multilayer Perceptron

Artificial neural networks, also called Multilayer Perceptron (MP) or neural networks, are a broad field, where perceptron is a model of a neuron that serves as the base for larger neural networks. They are basic computational units with weighted input signals and an activation function to produce an output signal. An activation function is a mapping from the summed weighted input to the output and determines the threshold for neuron activation and the strength of the output signal. The lowest layer, also known as the visible layer, receives input from the dataset. Hidden layers are considered deep due to their previously slow training times; however, in recent years, due to advancements, this layer can be trained fast within seconds. The output layer generates a value or vector of values that are in line with the specifications needed to solve the problem (Brownlee, 2022).


## Results and Discussion

In this section, the results of the hybrid approach of sentiment analysis are discussed.


### Evaluating sentiment lexicon effectiveness

This study aims to investigate the sentiment labeling of 21,966 unannotated Amazon iPhone reviews using a rule-based approach, TextBlob and VADER. The reviews are annotated as positive when the sentiment score is +1, negative when the sentiment score is -1, and neutral when the sentiment score is 0. Figure 11 shows the number of reviews assigned to each sentiment label.

Figure 11: Number of instances of label for each sentiment.

![image](https://github.com/user-attachments/assets/ef6477a8-c3a3-43ed-8212-64072a744507)

After cleaning the data, in the Rating (21%) given on the Amazon website, negative values are ~4000 counts when compared with TextBlob (11%) and VADER (11.8%). The scores from TextBlob follow a standardized order in contrast to the Rating, resulting in over 14,000 positive sentiments, exceeding VADER. VADER looks at how strong and whether a term is positive or negative and gives a score to the words, where neutral values (12.2%) are higher when compared to the other two.


### Performance analysis of lexicon-based approaches

To find the performance of rule-based approaches, three sentiment labels (positive, negative, and neutral) are converted to binary (positive and negative). The performance matrices namely accuracy, precision, recall, and F1-score are evaluated, where TextBlob achieves the better results 76.5%, 79.93%, 89.2%, and 84.31%, respectively, when compared to VADER (66.51%, 74.53%, 80%, and 77.19%) (Figure 12).

Figure 12: Performance of TextBlob and VADER.

![image](https://github.com/user-attachments/assets/b7329653-9a4f-4d06-be3a-ed6005517e4f)


### Topic modeling using LDA and LSA

Topic modeling identifies prominent topics, in this case, based on TextBlob as it shows better accuracy. Finding the right number of topics and using an effective text preprocessing technique are critical to achieving topics that are meaningful, well-defined, and easy to understand (Prabhakaran, n.d.).


#### Latent Semantic Analysis (LSA)

In LSA, term weights are used which can be positive or negative, where positive values suggest that the terms contribute positively to the representation of the underlying semantic structure of the topic and negative values suggest that the terms contribute negatively. A large positive or negative value indicates a strong correlation. From Figure 13, it is noted that the term “work” has a strong positive correlation, and “screen” has a strong negative correlation in Topic 1.

Figure 13: LSA topic modeling.

![image](https://github.com/user-attachments/assets/a6313d67-0f65-45a6-a3a3-300e055d5adf)

From LSA topics 1-5, positive and negative terms are segregated in Table 11 and related topic names are provided.

Table 11: LSA for topics 1-5 with positive and negative terms.

![image](https://github.com/user-attachments/assets/b86c0f3c-e7ef-417c-9bbd-f618e5102da7)
![image](https://github.com/user-attachments/assets/d6e42da4-a8ba-432f-8285-66d5ad3f058f)

Topic 1: Device satisfaction
Users employ positive terms, i.e., like and great, when discussing about device usage. However, there are also negative terms associated with concerns about purchasing a new device and about the screen.

Topic 2: iPhone longevity
Consumers express positive sentiments about the duration of device usage. In contrast, negative sentiments indicate potential concerns related to purchasing a new device. 

Topic 3: Device esthetics and replacement
Customers emote positive sentiments regarding the appearance and functionality of the devices. Nevertheless, negative sentiments are associated with the replacement of old iPhones with new ones.

Topic 4: Apple iPhone brand
In this topic, users are more positive about the brand, condition, and protectors of the device. However, negative sentiments are raised due to battery issues and scratches on the screen.

Topic 5: Overall satisfaction
Although customers are highlighting the positive aspects of the overall device experience, there are negative comments associated with considerations related to iPhone esthetics, battery, and possible replacement of new devices.

On one hand, users are positive about iPhone device satisfaction, longevity, esthetics, and brand; on the other hand, negative reviews emerge regarding purchase, screen and battery problems, and replacement.


#### Latent Dirichlet Allocation (LDA)

The words associated with each topic are visualized using the Gensim library; an intertopic distance map is displayed using the pyLDAvis—an interactive chart—for visualization. The tokenized terms are used by converting to BOW representation. For effective topic modeling, hyperparameters are optimized, where the number of topics is set to 15, the random size state is 100, the chunksize is 1000, and the number of passes is 50. The top 30 most relevant terms in all tokenized terms for Topic 1 are given in Figure 13, where the term ‘work’ has more than 1500 occurrences. In Figure 13, on the left, each bubble denotes a topic. The bigger bubbles imply that the topics are more prominent. More number of topics lead to many overlaps and form small-sized bubbles grouped in a specific region (Figure 13).

Figure 13: Topic 1: Top 30 most relevant terms related to positive context.

![image](https://github.com/user-attachments/assets/d603540d-5e8b-495f-9fbb-de5bff7c6dbc)

From Figure 13, the top 30 most relevant themes related to the positive context given in topics 1-5 are tabulated in Table 12.

Table 12: Top 30 most relevant terms related to positive context in topics 1-5.

![image](https://github.com/user-attachments/assets/d4a6dcd9-215d-478c-af35-20052df527de)
![image](https://github.com/user-attachments/assets/dc2df18d-e6c0-4b9a-9785-4c4bedbfd720)










Breuss
Shah
Pascual
Shivanandhan
Brownlee
Somarouthu
Karabiber
Khetan
Dube
Gomede
Raj
IBM
Donges
Brownlee
Prabhakaran
