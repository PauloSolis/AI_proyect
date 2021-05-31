from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_results(predicted_data, true_data, title='', xlab='', ylab=''):
    plt.title(title)
    #plt.plot(range(len(predicted_data)), predicted_data, label='Prediction')
    #print('Data set', range(len(true_data)))
    plt.plot(range(len(true_data)), true_data, label='True Data')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.show()
    return


df = pd.read_csv('dataset.csv')

# Show the data.
df
# Variable for predictiong ndays in future
days = 14

# New column where prediction will be stored
df['Prediction'] = df[['Close']].shift(-days)

# Show the dataset
print('Nueva tabla')
df.tail(7)

# New Data set
X = np.array(df[['Close']])
# Remove the last n rows

X = X[:-days]
print('Data set without last n Rows \n', X)

#
y = df['Prediction'].values
y = y[:-days]
print('Este es y \n',y)

# Split the data 85 training 15% test

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.15)

# Create and train the model
Reg = LinearRegression()
# Train the model
Reg.fit(x_train, y_train)

# Variable to be equal to the last n data from the orriginal data set

x_ = np.array(df[['Close']])[-days:]
print(x_)

# Print regrsion models
linear_prediction = Reg.predict(x_)
print(linear_prediction)
acc = cross_val_score(Reg, X, y)
print('Variance score: %.2f' % Reg.score(x_train, y_train))
print('Mean Cross Validation Score: %.2f' % -np.mean(acc))

plot_results(linear_prediction, X, 'Bitcoin price', 'Days', 'USD')
