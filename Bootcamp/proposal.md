## Objective:  
Identify alternate groupings of AGU talks/posters that might facilitate researchers exposing themselves to characterizations of problems similar to those they work on in some way, but which are in sections they would not traditionally peruse.  

## Value: 
As a scientist gets deeper in their career, there is a tendency to use a mix of broad and refined filters to filter out presentations that would be a waste of resources. This is not unreasonable because no human has the mental capacity to individually analyze over 10,000 abstracts. However, these heuristics also lead to presentations of potential value being passed over. By offering an alternative organizational scheme, scientists might consider branching out, in turn potentially reducing the amount of work that is redone because of topic isolation.

## Data:  
American Geophysical Union abstracts from one or more of the last fall meetings (10k+ documents with key words, authors, titles, session names and section designations)

## Approach:  
I anticipate pulling the data by web scraping and cleaning using standard parsing methods in preparation for feature creation using doc2vec and LSA.  I plan to cluster the data at various scales with various algorithms  (likely first pass clustering, then clustering the clusters) and doing some by hand evaluation as well as some assessment of the session, title and key word assemblages produced.  As a follow up, I will retain the cluster labels and attempt to reverse the system by training supervised classifiers to predict the cluster labels and analyzing the fit.  Additionally, I plan to train supervised classifiers to label according to section and keyword.  

In order to assess and present my work, I plan to offer a variety of basic statistics and counts about the corpus, confusion matrices, comparisons of session content, silhouette plots (and related statistics), TSNE plots, and perhaps network visualizations of keyword closeness.   

## Challenges:  
Achieving a big enough, but still tractable cluster size. 
