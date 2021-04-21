# CSE3521FinalProject
- We plan on solving the problem of figuring out trends in companies stocks.  We plan on taking the data from the companies on the NYSE as well as data pertaining to those companies on social media.  We will use this data to see trends and predict the price changes for those companies.
- References
    - https://finance.yahoo.com/quote/FB/history?guccounter=1&guce_referrer=aHR0cHM6Ly9kdWNrZHVja2dvLmNvbS8&guce_referrer_sig=AQAAAFxzXK9K7J8mbd6BJArTIqLEjf7aHLNt04Ha85bCO_j80mBSdIOLzXVTtCa-0F-MGuAQzoeMldDhJMAkCEvl_1zPW37X62JBC6U3lSIpf8qdPJOUbPRE36xUMtm9EX-vp8x2_1UJzN_-bMg6EzOxbJyD_TttSAdQmdQNPuIJdPTP
    - https://blog.quantinsti.com/machine-learning-logistic-regression-python/
## **Required Libraries**
1. NumPy
2. Pandas
3. TA-Lib (Graphing library)
4. Matplotlib
5. sklearn (For comparison only)
6. yFinance (This library can download data right from the website)


## **Execution**
- To run the ID3 algortihm on with the input data from the Data folder use: `python3 ID3.py`
- To run the logistic regression algorithm on the data use: `python3 Linear_Classifier.py`
- To run any file named "LRTest"name".py" use: `python3 LRTest"name".py`

## **Results** 
1. Our version of the ID3 algortihm with TSLA, FB, and AAPL:
   - FB accuracy: 0.52
   - AAPL accuracy: 0.45
   - TSLA accuracy: 0.46
2. Our version of the logistic regression algorithm
   - Accuracy: training set:  0.4857142857142857
   - Accuracy: test set:  0.49765258215962443
3. "Off the shelf" implementation of ID3:
   - **TO DO**
4. "Off the shelf" implementation of logistic regression:
    - AAPL accuracy: 0.53
    - TSLA accuracy: 0.5
    - FB accuracy: 0.55

## **TO-DO**
1. ~~Possibly download more data from various companies and modify the files to look like the current train and test data.~~
2. ~~Keep testing to see if the current implementation of the ID3 algorithm is producing reasonable results.~~
3. Use a 3rd party library implementation to compare the results.
4. Document all results and include full description of project in the Google Doc.
5. Turn in 1 day early (4/29)

