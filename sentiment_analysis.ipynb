# sentiment analysis
import csv

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
from xgboost import XGBClassifier
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ClassPredictionError

nltk.download('stopwords')

stopwordsALB = ['e', 'te', 'të', 'i', 'me', 'më', 'qe', 'që', 'në', 'një', 'a', 'për', 'sh', 'nga', 'ka', 'u', 'është',
                'dhe', 'shih', 'nuk', 'm', 'diçka', 'ose', 'si', 'shumë', 'etj', 'se', 'pa', 'sipas', 's', 't',
                'dikujt', 'dikë', 'mirë', 'vet', 'bëj', 'ai', 'vend', 'prej', 'ja', 'duke', 'tjetër', 'kur', 'ia',
                'ku', 'ta', 'keq', 'dy', 'bën', 'bërë', 'bëhet', 'diçkaje', 'edhe', 'madhe', 'la', 'sa', 'gjatë',
                'zakonisht', 'pas', 'veta', 'mbi', 'disa', 'iu', 'mos', 'c', 'para', 'dikush', 'gjë', 'bë', 'pak',
                'tek', 'fare', 'bëri', 'po', 'bie', 'k', 'do', 'gjithë', 'vetë', 'mund', 'kam', 'le', 'jo', 'bëje',
                'tij', 'kanë', 'ishte', 'janë', 'vjen', 'atë', 'ketë', 'nëpër', 'na', 'marrë', 'merr', 'mori', 'rri',
                'deri', 'b', 'kishte', 'mban', 'përpara', 'tyre', 'marr', 'gjitha', 'as', 'vetëm', 'nën', 'herë',
                'tjera', 'tjerët', 'drejt', 'qenët', 'ndonjë', 'nëse', 'jap', 'merret', 'rreth', 'lloj', 'dot', 'saj',
                'ndër', 'ndërsa', 'cila', 'veten', 'vetën', 'ma', 'ndaj', 'mes', 'ajo', 'cilën', 'por', 'ndërmjet',
                'prapa', 'mi', 'tërë', 'jam', 'ashtu', 'kësaj', 'tillë', 'bëhem', 'cilat', 'kjo', 'menjëherë', 'ca',
                'je', 'aq', 'aty', 'pranë', 'ato', 'pasur', 'qenë', 'cilin', 'tepër', 'njëra', 'tej', 'krejt', 'kush',
                'bëjnë', 'ti', 'bënë', 'midis', 'cili', 'ende', 'këto', 'kemi', 'siç', 'kryer', 'cilit', 'atij',
                'gjithnjë', 'andej', 'sipër', 'sikur', 'këtej', 'cilës', 'ky', 'papritur', 'ua', 'kryesisht',
                'gjithcka', 'pasi', 'kryhet', 'mjaft', 'këtij', 'përbashkët', 'ata', 'atje', 'vazhdimisht', 'kurrë',
                'tonë', 'kështu', 'unë', 'sapo', 'rrallë', 'vetes', 'vetës', 'ishin', 'afërt', 'tjetrën', 'këtu',
                'çfarë', 'to', 'anës', 'jemi', 'asaj', 'secila', 'kundrejt', 'këtyre', 'pse', 'tilla', 'mua',
                'nëpërmjet', 'cilët', 'ndryshe', 'kishin', 'ju', 'tani', 'atyre', 'diç', 'ynë', 'kudo', 'sonë',
                'sepse', 'cilave', 'kem', 'ty', 'duhet', 'apo', 'këtë', 'jetë', 'parë', 'çdo', 'ne', 'nje', 'per',
                'eshte', 'dicka', 'shume', 'dike', 'mire', 'bej', 'tjeter', 'ben', 'bere', 'behet', 'dickaje', 'gjate',
                'gje', 'be', 'beri', 'gjithe', 'vete', 'beje', 'kane', 'jane', 'ate', 'kete', 'neper', 'cdo', 'marre',
                'perpara', 'vetem', 'nen', 'here', 'tjeret', 'qenet', 'ndonje', 'nese', 'nder', 'ndersa', 'cilen',
                'ndermjet', 'tere', 'kesaj', 'tille', 'behem', 'menjehere', 'prane', 'qene', 'teper', 'njera', 'bejne',
                'bene', 'keto', 'sic', 'gjithnje', 'siper', 'ketej', 'ciles', 'ketij', 'perbashket', 'kurre', 'tone',
                'keshtu', 'une', 'rralle', 'afert', 'tjetren', 'ketu', 'cfare', 'anes', 'ketyre', 'nepermjet', 'cilet',
                'dic', 'yne', 'sone', 'asnjë', 'asnje', 'jeni', 'juaj', 'kaq', 'ke', 'mu', 'së', 'seç', 'sec', 'tëte',
                'tuaj', 'veç', 'vec', 'pra', 'atëherë', 'duket', 'tash', 'thotë', 'madje', 'sot', 'sesa', 'çka',
                'këta', 'imja', 'imi', 'jona', 'joni', 'juaja', 'jotja', 'joti', 'vetja', 'tija', 'saja', 'cilëve',
                'kujt', 'kë', 'isha', 'ishe', 'kisha', 'bëra', 'bënte', 'derisa', 'kundër', 'brenda', 'përmes', 'lart',
                'poshtë', 'jashtë', 'ulje', 'përfundoi', 'fund', 'përsëri', 'tutje', 'njëherë', 'gjithave',
                'gjithëçka', 'gjithçka', 'dyja', 'secili', 'çdonjëri', 'shumica', 'tjerave', 'njëjta', 'njëjtë',
                'gjithashtu', 'don', 'tjerë', 're', 'bëjë', 'jenë', 'pikërisht', 'askush', 'ndokush', 'ndonjëra',
                'duhej', 'ri', 'pastaj', 'andaj', 'ndoshta', 'tha', 'qoftë', 'asgjë', 'di', 'thënë', 'madh', 'qartë',
                'tashmë', 'sidomos', 'mëdha', 'lartë', 'reja', 'dhënë', 'prandaj', 'shkak', 'the', "t'i", 'nbsp'];


# stopwordsEN = stopwords.words('english')


# vectorization (vectorization involves representing text documents as numerical vectors, convert textual data into numerical format while preserving important information about the text)
# TF-IDF is similar to BoW but takes into account the importance of words in a document relative to their importance in the entire corpus. It assigns higher weights to words that are frequent in a document but rare in the corpus.
def tfidf_vectorizer(X, stopwords):
    if stopwords: # without stopwords
        tfidf_vect = TfidfVectorizer(analyzer="word", stop_words=stopwordsALB) # max_features=3000, default of analyzer = "word", stop_words=stopwordsALB, all of stopwords will be removed from the resulting tokens
    else:
        tfidf_vect = TfidfVectorizer(analyzer="word")  # max_features=3000, default of analyzer = "word"
    X_tfidf = tfidf_vect.fit_transform(X)#.toarray()
    return tfidf_vect, X_tfidf


# vectorization (vectorization involves representing text documents as numerical vectors, convert textual data into numerical format while preserving important information about the text)
# This method represents text data by counting the frequency of each word in the document. Each document is represented as a vector, where each element corresponds to the frequency of a particular word.
def count_vectorizer(X, stopwords):
    if stopwords: # without stopwords
        count_vect = CountVectorizer(analyzer="word", stop_words=stopwordsALB) # max_features=3000, default of analyzer = "word", stop_words=stopwordsALB, all of stopwords will be removed from the resulting tokens
    else:
        count_vect = CountVectorizer(analyzer="word")  # max_features=3000, default of analyzer = "word"
    X_count = count_vect.fit_transform(X)
    return count_vect, X_count


def confusion_matrix_yellowbrick(classifier, X_train, y_train, X_test, y_test, classes):
    cm = ConfusionMatrix(classifier, classes=classes)

    # fits the passed model
    cm.fit(X_train, y_train)

    # some test data. Score runs predict() on the data and then creates the confusion_matrix
    cm.score(X_test, y_test)

    # visualize
    cm.show()


def class_prediction_error_yellowbrick(classifier, X_train, y_train, X_test, y_test, classes):
    # instantiate the classification model and visualizer
    visualizer = ClassPredictionError(classifier, classes=classes)

    # fit the training data to the visualizer
    visualizer.fit(X_train, y_train)

    # evaluate the model on the test data
    visualizer.score(X_test, y_test)

    # visualize
    visualizer.show()


def feature_importance_scores(classifier, vect, top):
    # get feature coefficients
    coefficients = classifier.coef_[0]

    # get feature names from the vectorizer's vocabulary
    feature_names = vect.get_feature_names_out()

    # sort feature coefficients and feature names in descending order
    indices = coefficients.argsort()[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_coefficients = coefficients[indices]

    # select top features
    top_feature_names = sorted_feature_names[:top]
    top_coefficients = sorted_coefficients[:top]

    plt_title = 'Top ' + str(top) + ' Feature Coefficients'

    # plot the top feature coefficients vertically with thicker bars
    plt.figure(figsize=(10, 8))
    plt.barh(top_feature_names, top_coefficients, height=0.5)
    plt.xlabel('Coefficient')
    plt.ylabel('Features')
    plt.title(plt_title)
    plt.gca().invert_yaxis()  # invert y-axis to display the most important features at the top
    plt.show()


def get_data(file_path, column_name, key_column_name):
    df = pd.read_csv(file_path)
    df.dropna(subset=[column_name, key_column_name], inplace=True)
    label_encoder = LabelEncoder()
    df[key_column_name] = label_encoder.fit_transform(df[key_column_name])
    X = df[column_name]  # Features
    y = df[key_column_name]  # Target variable

    return X, y


def train_experiments(feature, stopwords, classifier):
    # example usage
    from google.colab import drive
    drive.mount('/content/drive')

    path = '/content/drive/MyDrive/Colab Notebooks/'

    file_path = path + 'sentiment_analysis/dataset_new.csv'
    column_name = 'Comment'
    key_column_name = 'ClassB'

    X, y = get_data(file_path=file_path, column_name=column_name, key_column_name=key_column_name)

    classes = list(set(y))
    if len(classes) == 2:
        classes = ["negative", "positive"]
    else:
        classes = ["neutral", "negative", "positive"]

    if feature == "tf-idf":
        vect, X_vect = tfidf_vectorizer(X, stopwords)
    elif feature == "bag-of-words":
        vect, X_vect = count_vectorizer(X, stopwords)

    # split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

    # train the model
    classifier.fit(X_train, y_train)

    # predict sentiment on the test set
    y_pred = classifier.predict(X_test)

    # evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # CROSS-VALIDATION
    scores = cross_val_score(classifier, X_vect, y, cv=5, scoring='f1_weighted')
    avg = sum(scores) / len(scores)
    print("scores", scores)
    print("avg score", avg)

    confusion_matrix_yellowbrick(classifier, X_train, y_train, X_test, y_test, classes)

    class_prediction_error_yellowbrick(classifier, X_train, y_train, X_test, y_test, classes)

    feature_importance_scores(classifier, vect, top=30)


features = ["tf-idf", "bag-of-words"]
stopwords = [True, False]
classifiers = [LogisticRegression(), LinearSVC(), SGDClassifier(), MultinomialNB(alpha=0.01), XGBClassifier()]
train_experiments(feature=features[0], stopwords=stopwords[0], classifier=classifiers[2])


