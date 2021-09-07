
## Introduction

This module helps you extract key terms and topics from corpus using a comparative approach.

## Installation

## Usage

### Import packages


```python
from compExtract import ComparativeExtraction
```

### Load sample data


```python
import pandas as pd
import numpy as np
PATH = "/Users/xiaoma/Desktop/gitrepo/associate-term-search/data/switch_reviews.csv"
data = pd.read_csv(PATH)
label = [x <= 3 for x in data['stars']]
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stars</th>
      <th>titles</th>
      <th>reviews</th>
      <th>dates</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>Worth It\n</td>
      <td>Definitely worth the money!\n</td>
      <td>September 21, 2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>Nintendo Swich gris joy con\n</td>
      <td>Con este producto no he sentido mucha satisfac...</td>
      <td>September 20, 2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>My kid wont put it down\n</td>
      <td>Couldnt of been happier, came early.  I was th...</td>
      <td>September 20, 2019</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>Happy\n</td>
      <td>Happy\n</td>
      <td>September 20, 2019</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>Great\n</td>
      <td>Great product\n</td>
      <td>September 19, 2019</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>1.0</td>
      <td>One Star\n</td>
      <td>it is no good, it suck, no work, plz hlp amazon\n</td>
      <td>December 12, 2017</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>5.0</td>
      <td>A must have gaming system\n</td>
      <td>The Nintendo Switch is a versatile hybrid game...</td>
      <td>December 12, 2017</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>5.0</td>
      <td>Switch\n</td>
      <td>This purchase save me from looking for one.\n</td>
      <td>December 11, 2017</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>5.0</td>
      <td>Five Stars\n</td>
      <td>Best babysitter ever!\n</td>
      <td>December 11, 2017</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>5.0</td>
      <td>Five Stars\n</td>
      <td>Its a great game console.\n</td>
      <td>December 11, 2017</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 4 columns</p>
</div>




```python
data.columns
```




    Index(['stars', 'titles', 'reviews', 'dates'], dtype='object')



Here we are using online Amazon reviews for Nintendo Switch to illustrate the usages of the module. 

The module requires a corpus and a set of binary labels as inputs. The labels should be created depending on what type of questions are we trying to answer. The set of labels should be of the same length as the corpus.

Here, let's assume that we want to know why people dislike this product and find relevant keywords. To answer this question, we created the label to be a binary variable indicating whether a reviewer gives a 3 star or less. 

### Initialize the module with the review corpus and labels


```python
ce = ComparativeExtraction(corpus = data['reviews'], labels = label)
```

### Extract the keywords


```python
ce.get_distinguish_terms(ngram_range = (1,3),top_n = 10)
```




    <compExtract.ComparativeExtraction at 0x7ff96f84b588>




```python
# Get the keywords that are mentioned significantly more in the less than or equal to 3 star reviews
ce.increased_terms_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>diff</th>
      <th>pos_prop</th>
      <th>pos_count</th>
      <th>neg_prop</th>
      <th>neg_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>work</td>
      <td>0.194976</td>
      <td>0.278426</td>
      <td>191</td>
      <td>0.083449</td>
      <td>360</td>
    </tr>
    <tr>
      <th>1</th>
      <td>switch</td>
      <td>0.176764</td>
      <td>0.351312</td>
      <td>241</td>
      <td>0.174548</td>
      <td>753</td>
    </tr>
    <tr>
      <th>2</th>
      <td>buy</td>
      <td>0.174520</td>
      <td>0.297376</td>
      <td>204</td>
      <td>0.122856</td>
      <td>530</td>
    </tr>
    <tr>
      <th>3</th>
      <td>month</td>
      <td>0.143129</td>
      <td>0.158892</td>
      <td>109</td>
      <td>0.015763</td>
      <td>68</td>
    </tr>
    <tr>
      <th>4</th>
      <td>nintendo</td>
      <td>0.134316</td>
      <td>0.290087</td>
      <td>199</td>
      <td>0.155772</td>
      <td>672</td>
    </tr>
    <tr>
      <th>5</th>
      <td>charge</td>
      <td>0.122855</td>
      <td>0.141399</td>
      <td>97</td>
      <td>0.018544</td>
      <td>80</td>
    </tr>
    <tr>
      <th>6</th>
      <td>use</td>
      <td>0.118448</td>
      <td>0.206997</td>
      <td>142</td>
      <td>0.088549</td>
      <td>382</td>
    </tr>
    <tr>
      <th>7</th>
      <td>new</td>
      <td>0.113989</td>
      <td>0.160350</td>
      <td>110</td>
      <td>0.046361</td>
      <td>200</td>
    </tr>
    <tr>
      <th>8</th>
      <td>would</td>
      <td>0.106540</td>
      <td>0.164723</td>
      <td>113</td>
      <td>0.058183</td>
      <td>251</td>
    </tr>
    <tr>
      <th>9</th>
      <td>get</td>
      <td>0.104055</td>
      <td>0.231778</td>
      <td>159</td>
      <td>0.127724</td>
      <td>551</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get the keywords that are mentioned significantly less in the less than or equal to 3 star reviews
ce.decreased_terms_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>diff</th>
      <th>pos_prop</th>
      <th>pos_count</th>
      <th>neg_prop</th>
      <th>neg_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>love</td>
      <td>-0.216997</td>
      <td>0.080175</td>
      <td>55</td>
      <td>0.297172</td>
      <td>1282</td>
    </tr>
    <tr>
      <th>1</th>
      <td>great</td>
      <td>-0.122247</td>
      <td>0.099125</td>
      <td>68</td>
      <td>0.221372</td>
      <td>955</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fun</td>
      <td>-0.048160</td>
      <td>0.046647</td>
      <td>32</td>
      <td>0.094808</td>
      <td>409</td>
    </tr>
    <tr>
      <th>3</th>
      <td>best</td>
      <td>-0.042638</td>
      <td>0.030612</td>
      <td>21</td>
      <td>0.073250</td>
      <td>316</td>
    </tr>
    <tr>
      <th>4</th>
      <td>amaze</td>
      <td>-0.038011</td>
      <td>0.010204</td>
      <td>7</td>
      <td>0.048215</td>
      <td>208</td>
    </tr>
    <tr>
      <th>5</th>
      <td>awesome</td>
      <td>-0.035827</td>
      <td>0.007289</td>
      <td>5</td>
      <td>0.043115</td>
      <td>186</td>
    </tr>
    <tr>
      <th>6</th>
      <td>son love</td>
      <td>-0.035564</td>
      <td>0.002915</td>
      <td>2</td>
      <td>0.038479</td>
      <td>166</td>
    </tr>
    <tr>
      <th>7</th>
      <td>perfect</td>
      <td>-0.032515</td>
      <td>0.008746</td>
      <td>6</td>
      <td>0.041261</td>
      <td>178</td>
    </tr>
    <tr>
      <th>8</th>
      <td>easy</td>
      <td>-0.026282</td>
      <td>0.023324</td>
      <td>16</td>
      <td>0.049606</td>
      <td>214</td>
    </tr>
    <tr>
      <th>9</th>
      <td>kid love</td>
      <td>-0.024370</td>
      <td>0.004373</td>
      <td>3</td>
      <td>0.028744</td>
      <td>124</td>
    </tr>
  </tbody>
</table>
</div>



If we need more context on a given word, or we need more interpretable topics, we can:
1. Output the reviews that contains the term
2. Switch the ngram_range

### Output the reviews

Say we want to know more about the significant term "work", we can directly output all the reviews containing the term.

The output class "kw" contains a one-hot encoded document-term-matrix that has all the terms found from the corpus. We can leverage it to find corresponding reviews of each term.


```python
# The binary_dtm provides a convenient way to extract reviews with specific terms
print(ce.binary_dtm[['work']])
```

          work
    0        0
    1        0
    2        0
    3        0
    4        0
    ...    ...
    4995     1
    4996     0
    4997     0
    4998     0
    4999     0
    
    [5000 rows x 1 columns]



```python
reviews_contain_term_work = data['reviews'][[x == 1 for x in ce.binary_dtm['work']]]
len(reviews_contain_term_work)
```




    551




```python
for x in pd.Series(reviews_contain_term_work).sample(1):
    print(x)
```

    I bought this as a Christmas present for my son.  After about a month and half of using it.  The switch stopped working.  It wont charge.  The product is an expensive piece of junk.
    


### Change the n-gram range to exclude uni-grams


```python
ce_ngram = ComparativeExtraction(corpus = data['reviews'], labels = label).get_distinguish_terms(ngram_range=(2,4), top_n=10)
```

    /Users/xiaoma/envs/compExtract/lib/python3.7/site-packages/sklearn/feature_extraction/text.py:489: UserWarning: The parameter 'token_pattern' will not be used since 'tokenizer' is not None'
      warnings.warn("The parameter 'token_pattern' will not be used"





    <compExtract.ComparativeExtraction at 0x7ff955f23cf8>




```python
ce_ngram.increased_terms_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>diff</th>
      <th>pos_prop</th>
      <th>pos_count</th>
      <th>neg_prop</th>
      <th>neg_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>joy con</td>
      <td>0.040857</td>
      <td>0.056851</td>
      <td>39</td>
      <td>0.015994</td>
      <td>69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>brand new</td>
      <td>0.020511</td>
      <td>0.027697</td>
      <td>19</td>
      <td>0.007186</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nintendo switch</td>
      <td>0.019638</td>
      <td>0.074344</td>
      <td>51</td>
      <td>0.054706</td>
      <td>236</td>
    </tr>
    <tr>
      <th>3</th>
      <td>buy switch</td>
      <td>0.018888</td>
      <td>0.027697</td>
      <td>19</td>
      <td>0.008809</td>
      <td>38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>play game</td>
      <td>0.014092</td>
      <td>0.039359</td>
      <td>27</td>
      <td>0.025267</td>
      <td>109</td>
    </tr>
    <tr>
      <th>5</th>
      <td>game play</td>
      <td>0.009812</td>
      <td>0.021866</td>
      <td>15</td>
      <td>0.012054</td>
      <td>52</td>
    </tr>
    <tr>
      <th>6</th>
      <td>year old</td>
      <td>0.005243</td>
      <td>0.023324</td>
      <td>16</td>
      <td>0.018081</td>
      <td>78</td>
    </tr>
    <tr>
      <th>7</th>
      <td>christmas gift</td>
      <td>0.003682</td>
      <td>0.014577</td>
      <td>10</td>
      <td>0.010895</td>
      <td>47</td>
    </tr>
    <tr>
      <th>8</th>
      <td>battery life</td>
      <td>0.001833</td>
      <td>0.024781</td>
      <td>17</td>
      <td>0.022949</td>
      <td>99</td>
    </tr>
    <tr>
      <th>9</th>
      <td>wii u</td>
      <td>0.000504</td>
      <td>0.016035</td>
      <td>11</td>
      <td>0.015531</td>
      <td>67</td>
    </tr>
  </tbody>
</table>
</div>




```python
ce_ngram.decreased_terms_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>diff</th>
      <th>pos_prop</th>
      <th>pos_count</th>
      <th>neg_prop</th>
      <th>neg_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>son love</td>
      <td>-0.035564</td>
      <td>0.002915</td>
      <td>2</td>
      <td>0.038479</td>
      <td>166</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kid love</td>
      <td>-0.024370</td>
      <td>0.004373</td>
      <td>3</td>
      <td>0.028744</td>
      <td>124</td>
    </tr>
    <tr>
      <th>2</th>
      <td>great game</td>
      <td>-0.018442</td>
      <td>0.007289</td>
      <td>5</td>
      <td>0.025730</td>
      <td>111</td>
    </tr>
    <tr>
      <th>3</th>
      <td>great product</td>
      <td>-0.014171</td>
      <td>0.004373</td>
      <td>3</td>
      <td>0.018544</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>great console</td>
      <td>-0.013641</td>
      <td>0.005831</td>
      <td>4</td>
      <td>0.019471</td>
      <td>84</td>
    </tr>
    <tr>
      <th>5</th>
      <td>best console</td>
      <td>-0.013609</td>
      <td>0.001458</td>
      <td>1</td>
      <td>0.015067</td>
      <td>65</td>
    </tr>
    <tr>
      <th>6</th>
      <td>highly recommend</td>
      <td>-0.012615</td>
      <td>0.002915</td>
      <td>2</td>
      <td>0.015531</td>
      <td>67</td>
    </tr>
    <tr>
      <th>7</th>
      <td>absolutely love</td>
      <td>-0.011987</td>
      <td>0.001458</td>
      <td>1</td>
      <td>0.013445</td>
      <td>58</td>
    </tr>
    <tr>
      <th>8</th>
      <td>game system</td>
      <td>-0.011746</td>
      <td>0.021866</td>
      <td>15</td>
      <td>0.033611</td>
      <td>145</td>
    </tr>
    <tr>
      <th>9</th>
      <td>love switch</td>
      <td>-0.011452</td>
      <td>0.013120</td>
      <td>9</td>
      <td>0.024571</td>
      <td>106</td>
    </tr>
  </tbody>
</table>
</div>


