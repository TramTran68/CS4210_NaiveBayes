#-------------------------------------------------------------------------
# AUTHOR: Tram Tran
# FILENAME: naive_bayes.py
# SPECIFICATION: Using Naive Bayes strategy, the program reads the file weather_training.csv (training set) 
# and output the classification of each of the 10 instances from the file weather_test (test set) if the classification confidence is >= 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
training_data = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            training_data.append(row)
            

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
outlook_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temp_map = {'Cool': 1, 'Mild': 2, 'Hot': 3}
humidity_map = {'High': 1, 'Normal': 2}
wind_map = {'Weak': 1, 'Strong': 2}

X = []
for data in training_data:
    X.append([outlook_map[data[1]], temp_map[data[2]], humidity_map[data[3]], wind_map[data[4]]])
    
#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
play_map = {'Yes': 1, 'No': 2}
Y = []
for data in training_data:
    Y.append(play_map[data[5]])

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
testing_data = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            testing_data.append(row)

#Printing the header of the solution
print(f"{'Day':<8} {'Outlook':<10} {'Temperature':<14} {'Humidity':<12} {'Wind':<8} {'PlayTennis':<10} {'Confidence':<10}")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for row in testing_data:
    # Convert the test data into numeric values using the mappings
    outlook_val = outlook_map[row[1]]
    temp_val = temp_map[row[2]]
    humidity_val = humidity_map[row[3]]
    wind_val = wind_map[row[4]]
    
    # Get the probabilities for the prediction
    probabilities = clf.predict_proba([[outlook_val, temp_val, humidity_val, wind_val]])[0]
    
    # Get the highest probability (confidence)
    confidence = max(probabilities)
    
    # Only process and print if confidence is >= 0.75
    if confidence >= 0.75:
        predicted_class = 'Yes' if probabilities[0] > probabilities[1] else 'No'
        # Printing the results in a formatted manner
        print(f"{row[0]:<8} {row[1]:<10} {row[2]:<14} {row[3]:<12} {row[4]:<8} {predicted_class:<10} {confidence:.2f}")


