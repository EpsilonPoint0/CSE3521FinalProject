from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

test_files = ["Data/FB.csv", "Data/AAPL.csv", "Data/TSLA.csv"]

for i in range(3):
    # Store the data into the df variable

    df = pd.read_csv(test_files[i])
    # Set the date as the index for the data
    df = df.set_index(pd.DatetimeIndex(df['Date'].values))
    # Give the index a name
    df.index.name = 'Date'

    # Manipulate the data set
    # Create the target column
    df['Price_Up'] = np.where(df['Close'].shift(-1) > df['Close'], 1,
                              0)  # if tomorrows price is greater than todays price put 1 else put 0
    # Remove the date column
    remove_list = ['Date']
    df = df.drop(columns=remove_list)

    X = df.iloc[:, 0:df.shape[1] - 1].values  # Get all the rows and columns except for the target column
    Y = df.iloc[:, df.shape[1] - 1].values  # Get all the rows from the target column

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    tree = DecisionTreeClassifier().fit(X_train, Y_train)

    print(f"Train ccuracy for {test_files[i]}: {tree.score(X_train, Y_train)}")

    print(f"Train ccuracy for {test_files[i]}: {tree.score(X_test, Y_test)}")

    tree_prediction = tree.predict(X_test)
