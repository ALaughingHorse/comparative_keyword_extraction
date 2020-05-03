"""
Last Modified: 05/03/2020
"""
import pandas as pd
import numpy as np
import nltk
nltk.download("wordnet")
from tokenizer_xm import text_tokenizer_xm, contractions
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LassoCV
from gensim.parsing.preprocessing import STOPWORDS
from imblearn.over_sampling import SMOTE
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import random

all_stops = list(STOPWORDS)
all_stops.remove("not")
all_stops.remove("no")

class comparative_keyword_extraction:
    """
    Takes in reviews with ratings. Outputs salient words that tells why people love/hate the product
    ---Parameters
    """
    
    def __init__(self, corpus, labels, stop_words = all_stops, addi_stops = []):
        
        self.corpus = corpus
        self.labels = labels
        self.stop_words = stop_words
        self.addi_stops = addi_stops


    def get_simple_keywords(self,ngram_range,min_df = 5, max_df = 0.9, product_names = []):
        """
        product_names: a list of unique product names
        """
        # Build a wrapper for tokenizer
        def tokenizer(text):
            tk = text_tokenizer_xm(text = text, lemma_flag = True, stem_flag = False,contractions=contractions)
            return tk.txt_pre_pros()

        # Get the simple word_count ranking
        ## adding product names as stop-words
        product_names = list(product_names)
        prod_names = " ".join(product_names)
        stop_words = text_tokenizer_xm(prod_names,lemma_flag=True, stem_flag = False,contractions = contractions).txt_pre_pros()
        stop_words += ['product']

        vec_count = CountVectorizer(ngram_range = ngram_range,tokenizer=tokenizer,min_df = min_df, max_df = max_df)
        vec_count_f = vec_count.fit(self.corpus)

        dtm = vec_count_f.transform(self.corpus)

        class output:
            terms = vec_count_f.get_feature_names()
            Document_Term_Matrix = pd.DataFrame(dtm.toarray())
            #rating_labels = self.ratings
        return output

    """
    init the vocab dict 
    """
    def update_dict(self, vocab,key,val):
        if key in vocab:
            vocab.update({key:vocab[key] + [val]})
        else:
            vocab.update({key:[val]})
    
    def tokenizer(self,text):
        stop_words = self.stop_words + self.addi_stops
        # Takes in the stopwords
        tk = text_tokenizer_xm(text = text, lemma_flag = True, stem_flag = False,contractions=contractions,stopwords = stop_words)
        return tk.txt_pre_pros()
        
    def get_popular_pos_tags(self,target_word, target_corpus):
        """
        get a target word and a list of text documents containing the word, return the most popular PoS tag of the word
        """
        # If a term is a N-gram, cannot match.
        if len(word_tokenize(target_word)) > 1:
            return "n-gram"

        def single_lemmatization(token):
            # A simple function for lemmatizing a single word.
            # This is necessary because the tagged tokens are not lemmatized for PoS performance
            # And we need to match words like "add" and "added"
            lemmatizer = WordNetLemmatizer()
            result = lemmatizer.lemmatize(token, pos='v')
            result = lemmatizer.lemmatize(result, pos='n')
            return result 
        # Pre-processing for PoS tagging
        simple_processed_doc = [word_tokenize(x.lower()) for x in target_corpus]
                        
        # Tag each word within a list of tokenized documents
        tagged_doc = pd.Series(simple_processed_doc).apply(lambda x: pos_tag(x))
        all_tags = []
        for docs in tagged_doc:
            # Get all the tags for target word
            target_tag = [tup[1] for tup in docs if  single_lemmatization(tup[0].lower()) == target_word.lower()]
            all_tags += target_tag

        tag_df = pd.DataFrame({'tag':all_tags})
        #get the most popular tag
        best_tags = tag_df.groupby('tag').size().sort_values(ascending = False).reset_index()['tag']

        if len(best_tags) == 0:
            return "None"
        else:
            return best_tags[0]

    def get_distinguishing_terms(self,ngram_range = (1,3), min_df = 0.01, max_df = 0.90,top_n = 20,tagging = False):
        
        """
        Get distinguishing terms based on term frequencies only without linear models
        """
        # a dataframe of all the reviews
        df = pd.DataFrame({"review_text":self.corpus,"labels":self.labels})

        # Fit the Count vectorizer
        vec_count = CountVectorizer(ngram_range = ngram_range,tokenizer=self.tokenizer,min_df =min_df, max_df = max_df,binary = True) 
        vec_count_f = vec_count.fit(df['review_text'])

        # Create the triaining document-term matrix
        dtm = vec_count_f.transform(df['review_text'])
        dtm_df = pd.DataFrame(dtm.toarray())
        dtm_df.columns = vec_count_f.get_feature_names()

        # Separate the terms into positive and negative dataframes
        pos_df = dtm_df[df['labels'] == 1].reset_index(drop = True)
        neg_df = dtm_df[df['labels'] == 0].reset_index(drop = True)

        # Get the proportional document frequencies for each term
        pos_shape = pos_df.shape[0]
        pos_count = [sum(pos_df[x] == 1) for x in vec_count_f.get_feature_names()]
        pos_prop = np.array(pos_count)/pos_shape

        neg_shape = neg_df.shape[0]
        neg_count = [sum(neg_df[x] == 1) for x in vec_count_f.get_feature_names()]
        neg_prop = np.array(neg_count)/neg_shape

        # Sort the table and filter by top N
        distinguish_df = pd.DataFrame({"feature":vec_count_f.get_feature_names(),\
            "diff":np.array(pos_prop) - np.array(neg_prop),\
            "pos_prop":pos_prop,\
            "pos_count":pos_count,\
            "neg_prop":neg_prop,\
            "neg_count":neg_count})

        # Sort the table in an ascending order and filter by top N
        increased_terms_df = distinguish_df.sort_values('diff',ascending = False).reset_index(drop = True).head(top_n)

        # Sort the table in an descending order and filter by top N
        decreased_terms_df = distinguish_df.sort_values('diff',ascending = True).reset_index(drop = True).head(top_n)

        # Get PoS tags
        if tagging:
            # Here we only tag the increased terms. Because this tagging is anticipated to only be used in Like vs Dislike, where the 
            # decreased terms are not used
            most_popular_tags = increased_terms_df['feature'].apply(lambda x: self.get_popular_pos_tags(x,self.corpus[dtm_df[x]]))
            increased_terms_df['PoS'] = most_popular_tags

        class output:
            incre_df = increased_terms_df
            decline_df = decreased_terms_df
            binary_dtm = dtm_df

        return output

    def get_keywords_with_regression(self,random_seed = 923,ngram_range = (1,1),min_df = 0.01, max_df = 0.85,apply_smote = True,\
        top_n = 25):
        
        """
        Get the keywords using LASSO's feature selection technique
        """
        # a dataframe of all the reviews
        df = pd.DataFrame({"review_text":self.corpus,"labels":self.labels})
        
        # Get positive Examples
        pos_df = df.loc[df['labels'] == 1,:].reset_index(drop = True)
        
        # Get negative examples
        neg_df = df.loc[df['labels'] == 0,:].reset_index(drop = True)

        # get the shape
        pos_count = pos_df.shape[0]
        neg_count = neg_df.shape[0]

        # Define the major vs minor dataframe
        minor_df = pos_df

        major_df = neg_df

        if pos_count > neg_count:
            minor_df = neg_df
            major_df = pos_df
            
        if minor_df.shape[0] < 100:
            print('Warning: Number of minority class less than 100')
        
        # initialize the variables
        #salient_terms = dict()

        # removed the iteration
        df_balanced = minor_df.append(major_df)
        df_balanced.reset_index(drop = True, inplace = True)
        target = df_balanced['labels']

        # Fit the TFidf vectorizer
        vec_count = TfidfVectorizer(ngram_range = ngram_range,tokenizer=self.tokenizer,min_df =min_df, max_df = max_df) 
        vec_count_f = vec_count.fit(df_balanced['review_text'])

        # Create the triaining document-term matrix
        vec_f = vec_count_f
        train_dtm = vec_f.transform(df_balanced['review_text'])

        if apply_smote:
             # Apply SMOTE to fix the class imbalance problem
            sm = SMOTE(random_state = 923,ratio = 1)

            X_input, y_input = sm.fit_resample(train_dtm,target)
        else:
            X_input, y_input = train_dtm, target

        """
        Modeling
        """
        lasso = LassoCV(cv = 5)
        lasso_f = lasso.fit(X_input,y_input)
        
        """
        Construct the coeficient table
        """
        coef_table = pd.DataFrame({"feature":vec_f.get_feature_names(),"coef":lasso_f.coef_})

        # Select only the positive coefficients
        key_terms_df = coef_table[['feature','coef']][coef_table['coef'] > 0]
        
        key_terms_df.reset_index(drop = True, inplace = True)

        """
        Next find the DF for the terms 
        """
        # Fit the Count vectorizer to the entire data
        vec_count = CountVectorizer(ngram_range = ngram_range,tokenizer=self.tokenizer,min_df = min_df, max_df = max_df)
        vec_count_f = vec_count.fit(df['review_text'].astype(str))

        # Create the triaining document-term matrix
        dtm = vec_count_f.transform(df['review_text'].astype(str))
        # Construct dataframe
        dtm_df = pd.DataFrame(dtm.toarray())
        dtm_df.columns = vec_count_f.get_feature_names()

        dtm_df_output = dtm_df.copy()
        dtm_key_terms = dtm_df[list(key_terms_df['feature'])]
        
        frequency_count = dtm_key_terms.sum(axis = 0)
        document_frequency = dtm_key_terms.apply(lambda x: sum(x != 0),axis = 0)

        # Create a dict to record the indexes
        index_tb = dtm_key_terms.apply(lambda x: list(x != 0),axis = 0)
        index_dict = {}
        for term in dtm_key_terms.columns:
            index_dict.update({term:np.array(index_tb[term].values)})
        self.index_dict = index_dict
        
        # Append columns 
        key_terms_df['Document_Count'] = document_frequency.values
        key_terms_df['Document_Percentage'] = key_terms_df['Document_Count']/len(df['review_text'])
        
        sorted_key_terms = key_terms_df.sort_values(["coef","Document_Count"],ascending = [False,False]).head(top_n)
        """
        As a last step, let's tag each term and separate into Nouns, Verbs and Adjs
        """
        most_popular_tags = sorted_key_terms['feature'].apply(lambda x: self.get_popular_pos_tags(x,self.corpus[index_dict[x]]))
        sorted_key_terms['PoS'] = most_popular_tags

        self.sorted_key_terms = sorted_key_terms
        
        class output:
            sorted_key_terms_tb = sorted_key_terms
            index_dictionary = index_dict
            dtm_count_df = dtm_df_output
            #salient_terms_dict = salient_terms
           
        return output


    ### Supplement functions for further information that can be extracted with the class above


