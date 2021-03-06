
## Introduction

This module helps you extract key terms and topics from corpus using a comparative approach.

## Installation

```
pip install --upgrade comparativeExtraction
```

## Usage

### Import packages


```python
from comparativeExtraction import comparative_keyword_extraction
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
data.columns
```




    Index(['stars', 'titles', 'reviews', 'dates'], dtype='object')



Here we are using online Amazon reviews for Nintendo Switch to illustrate the usages of the module. 

The module requires a corpus and a set of binary labels as inputs. The labels should be created depending on what type of questions are we trying to answer. The set of labels should be of the same length as the corpus.

Here, let's assume that we want to know why people dislike this product and find relevant keywords. To answer this question, we created the label to be a binary variable indicating whether a reviewer gives a 3 star or less. 

### Initialize the module with the review corpus and labels


```python
kw_init = comparative_keyword_extraction(corpus = data['reviews'], labels = label)
```

### Extract the keywords


```python
kw = kw_init.get_distinguishing_terms(ngram_range = (1,3),top_n = 10)
```


```python
# Get the keywords that are mentioned significantly more in the less than or equal to 3 star reviews
kw.incre_df
```




<div>
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
      <td>not</td>
      <td>0.345676</td>
      <td>0.481050</td>
      <td>330</td>
      <td>0.135373</td>
      <td>584</td>
    </tr>
    <tr>
      <th>1</th>
      <td>work</td>
      <td>0.202033</td>
      <td>0.285714</td>
      <td>196</td>
      <td>0.083681</td>
      <td>361</td>
    </tr>
    <tr>
      <th>2</th>
      <td>switch</td>
      <td>0.180442</td>
      <td>0.355685</td>
      <td>244</td>
      <td>0.175243</td>
      <td>756</td>
    </tr>
    <tr>
      <th>3</th>
      <td>buy</td>
      <td>0.172898</td>
      <td>0.297376</td>
      <td>204</td>
      <td>0.124478</td>
      <td>537</td>
    </tr>
    <tr>
      <th>4</th>
      <td>month</td>
      <td>0.141970</td>
      <td>0.158892</td>
      <td>109</td>
      <td>0.016922</td>
      <td>73</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nintendo</td>
      <td>0.138791</td>
      <td>0.301749</td>
      <td>207</td>
      <td>0.162958</td>
      <td>703</td>
    </tr>
    <tr>
      <th>6</th>
      <td>no</td>
      <td>0.131157</td>
      <td>0.190962</td>
      <td>131</td>
      <td>0.059805</td>
      <td>258</td>
    </tr>
    <tr>
      <th>7</th>
      <td>charge</td>
      <td>0.122458</td>
      <td>0.142857</td>
      <td>98</td>
      <td>0.020399</td>
      <td>88</td>
    </tr>
    <tr>
      <th>8</th>
      <td>use</td>
      <td>0.118979</td>
      <td>0.208455</td>
      <td>143</td>
      <td>0.089476</td>
      <td>386</td>
    </tr>
    <tr>
      <th>9</th>
      <td>new</td>
      <td>0.114520</td>
      <td>0.161808</td>
      <td>111</td>
      <td>0.047288</td>
      <td>204</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get the keywords that are mentioned significantly less in the less than or equal to 3 star reviews
kw.decline_df
```




<div>
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
      <td>-0.217692</td>
      <td>0.080175</td>
      <td>55</td>
      <td>0.297867</td>
      <td>1285</td>
    </tr>
    <tr>
      <th>1</th>
      <td>great</td>
      <td>-0.118801</td>
      <td>0.103499</td>
      <td>71</td>
      <td>0.222299</td>
      <td>959</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fun</td>
      <td>-0.048093</td>
      <td>0.048105</td>
      <td>33</td>
      <td>0.096198</td>
      <td>415</td>
    </tr>
    <tr>
      <th>3</th>
      <td>best</td>
      <td>-0.041875</td>
      <td>0.032070</td>
      <td>22</td>
      <td>0.073945</td>
      <td>319</td>
    </tr>
    <tr>
      <th>4</th>
      <td>amaze</td>
      <td>-0.039170</td>
      <td>0.010204</td>
      <td>7</td>
      <td>0.049374</td>
      <td>213</td>
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
      <td>-0.035033</td>
      <td>0.004373</td>
      <td>3</td>
      <td>0.039407</td>
      <td>170</td>
    </tr>
    <tr>
      <th>7</th>
      <td>perfect</td>
      <td>-0.032978</td>
      <td>0.008746</td>
      <td>6</td>
      <td>0.041725</td>
      <td>180</td>
    </tr>
    <tr>
      <th>8</th>
      <td>easy</td>
      <td>-0.026514</td>
      <td>0.023324</td>
      <td>16</td>
      <td>0.049838</td>
      <td>215</td>
    </tr>
    <tr>
      <th>9</th>
      <td>kid love</td>
      <td>-0.025066</td>
      <td>0.004373</td>
      <td>3</td>
      <td>0.029439</td>
      <td>127</td>
    </tr>
  </tbody>
</table>
</div>



If we need more context on a given word, or we need more interpretable topics, we can:
1. Output the reviews that contains the term
2. Switch the ngram_range
3. Use the supplement functions module 

### Output the reviews

Say we want to know more about the significant term "work", we can directly output all the reviews containing the term.

The output class "kw" contains a one-hot encoded document-term-matrix that has all the terms found from the corpus. We can leverage it to find corresponding reviews of each term.


```python
# The binary_dtm provides a convenient way to extract reviews with specific terms
print(kw.binary_dtm[['work','not']])
```

          work  not
    0        0    0
    1        0    0
    2        0    0
    3        0    0
    4        0    0
    ...    ...  ...
    4995     1    0
    4996     0    1
    4997     0    0
    4998     0    0
    4999     0    0
    
    [5000 rows x 2 columns]



```python
reviews_contain_term_work = data['reviews'][[x == 1 for x in kw.binary_dtm['work']]]
len(reviews_contain_term_work)
```




    557




```python
for x in pd.Series(reviews_contain_term_work).sample(1):
    print(x)
```

    It's alright, only got it to give Nintendo another chance. It's a neat concept. Overall, it's aggressively mediocre, good for casual stuff, but will never get as much use as my ps4.Wi-Fi is god awful though. The worst I've dealt with. It's connection capabilities are atrocious compared with any other wireless device. Don't expect it to just work. Honestly, this singular problem is enough for me to rate it 1 star. I suppose they had to cut corners somewhere.
    


### Change the n-gram range to exclude uni-grams


```python
kw = kw_init.get_distinguishing_terms(ngram_range = (2,4),top_n = 10)
kw.incre_df
```




<div>
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
      <td>not work</td>
      <td>0.077393</td>
      <td>0.080175</td>
      <td>55</td>
      <td>0.002782</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>buy switch</td>
      <td>0.022566</td>
      <td>0.032070</td>
      <td>22</td>
      <td>0.009504</td>
      <td>41</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nintendo switch</td>
      <td>0.021395</td>
      <td>0.077259</td>
      <td>53</td>
      <td>0.055865</td>
      <td>241</td>
    </tr>
    <tr>
      <th>3</th>
      <td>brand new</td>
      <td>0.020279</td>
      <td>0.027697</td>
      <td>19</td>
      <td>0.007418</td>
      <td>32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>play game</td>
      <td>0.014921</td>
      <td>0.042274</td>
      <td>29</td>
      <td>0.027353</td>
      <td>118</td>
    </tr>
    <tr>
      <th>5</th>
      <td>game not</td>
      <td>0.012563</td>
      <td>0.026239</td>
      <td>18</td>
      <td>0.013676</td>
      <td>59</td>
    </tr>
    <tr>
      <th>6</th>
      <td>year old</td>
      <td>0.009683</td>
      <td>0.029155</td>
      <td>20</td>
      <td>0.019471</td>
      <td>84</td>
    </tr>
    <tr>
      <th>7</th>
      <td>game play</td>
      <td>0.005176</td>
      <td>0.021866</td>
      <td>15</td>
      <td>0.016690</td>
      <td>72</td>
    </tr>
    <tr>
      <th>8</th>
      <td>nintendo game</td>
      <td>0.003616</td>
      <td>0.013120</td>
      <td>9</td>
      <td>0.009504</td>
      <td>41</td>
    </tr>
    <tr>
      <th>9</th>
      <td>christmas gift</td>
      <td>0.003451</td>
      <td>0.014577</td>
      <td>10</td>
      <td>0.011127</td>
      <td>48</td>
    </tr>
  </tbody>
</table>
</div>




```python
kw.decline_df
```




<div>
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
      <td>-0.035033</td>
      <td>0.004373</td>
      <td>3</td>
      <td>0.039407</td>
      <td>170</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kid love</td>
      <td>-0.025066</td>
      <td>0.004373</td>
      <td>3</td>
      <td>0.029439</td>
      <td>127</td>
    </tr>
    <tr>
      <th>2</th>
      <td>great game</td>
      <td>-0.020760</td>
      <td>0.007289</td>
      <td>5</td>
      <td>0.028048</td>
      <td>121</td>
    </tr>
    <tr>
      <th>3</th>
      <td>great product</td>
      <td>-0.014403</td>
      <td>0.004373</td>
      <td>3</td>
      <td>0.018776</td>
      <td>81</td>
    </tr>
    <tr>
      <th>4</th>
      <td>great console</td>
      <td>-0.014104</td>
      <td>0.005831</td>
      <td>4</td>
      <td>0.019935</td>
      <td>86</td>
    </tr>
    <tr>
      <th>5</th>
      <td>highly recommend</td>
      <td>-0.013311</td>
      <td>0.002915</td>
      <td>2</td>
      <td>0.016226</td>
      <td>70</td>
    </tr>
    <tr>
      <th>6</th>
      <td>absolutely love</td>
      <td>-0.012219</td>
      <td>0.001458</td>
      <td>1</td>
      <td>0.013676</td>
      <td>59</td>
    </tr>
    <tr>
      <th>7</th>
      <td>love switch</td>
      <td>-0.012147</td>
      <td>0.013120</td>
      <td>9</td>
      <td>0.025267</td>
      <td>109</td>
    </tr>
    <tr>
      <th>8</th>
      <td>best console</td>
      <td>-0.010926</td>
      <td>0.004373</td>
      <td>3</td>
      <td>0.015299</td>
      <td>66</td>
    </tr>
    <tr>
      <th>9</th>
      <td>best game</td>
      <td>-0.010297</td>
      <td>0.002915</td>
      <td>2</td>
      <td>0.013213</td>
      <td>57</td>
    </tr>
  </tbody>
</table>
</div>



### Using supplement function

Sometimes when we want to drill down into one specific term, we can leverage the built-in supplement functions to find related n-grams containing the term


```python
from comparativeExtraction.supplement_funcs import get_ngrams_on_term
```


```python
target_term = "work"
reviews_contain_term_work = data['reviews'][[x == 1 for x in kw.binary_dtm['work']]]

related_ngrams = get_ngrams_on_term(target_term,reviews_contain_term_work,filter_by_extreme=False)
```


```python
related_ngrams.related_ngrams.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ngram</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>work great</td>
      <td>104</td>
    </tr>
    <tr>
      <th>1</th>
      <td>not work</td>
      <td>71</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stop work</td>
      <td>47</td>
    </tr>
    <tr>
      <th>3</th>
      <td>work fine</td>
      <td>38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>work perfectly</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>



Here, the count is also a Document Frequency
