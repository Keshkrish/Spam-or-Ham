import numpy as np
import os
import pandas as pd
from sklearn import svm
from sklearn import metrics
import joblib

def Trainer():
    df = pd.read_csv("enron_spam_data.csv", sep=',')  # dataset is imported

    total_dataset = df.to_numpy()
    # The first feature denotes the mail number
    # second feature stores the subject as a string
    # third feature stores the body of the email
    # fourth feature indicates if the email is spam or not
    # fifth feature stores the date of the email(it is ignored here)
    np.random.shuffle(total_dataset)  # dataset is shuffled
    y = []  # The ith element of this array is 0 if the ith mail after shuffling is non spam,1 otherwise
    for i in range(len(total_dataset)):
        if total_dataset[i, 3] == 'ham':
            y.append(0)
        elif total_dataset[i, 3] == 'spam':
            y.append(1)

    dataset = total_dataset[
              0:10000]  # only the first 10000 mails are used for training to limit the time and space required

    all_words = {}  # a dictionary which stores all the distinct words found in the dataset, each word is assigned a unique number
    # This number denotes the feature in the email vector which stores the information of this word
    b = 0
    dataset_words = [
        []]  # Stores all the emails. The ith element in this is a list containing the words in the ith mail in the same order as encountered in the email

    for i in range(len(dataset)):

        if pd.isnull(dataset[i, 1]):  # To check if the subject is empty
            dataset[i, 1] = ''

        if pd.isnull(dataset[i, 2]):  # To check if the body is empty
            dataset[i, 2] = ''

        dataset[i, 1] = dataset[i, 1].lower()
        dataset[i, 2] = dataset[i, 2].lower()
        characters = ['~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '[', ']', '{', '}',
                      '\\', '|', '"', "'", ':', ';', ',', '<', '.', '>', '/', '?', '/', '*', '-', '+']
        # The above special characters are removed from the string
        for c in characters:
            dataset[i, 1] = dataset[i, 1].replace(c, '')
            dataset[i, 2] = dataset[i, 2].replace(c, '')
        suffix = ['ing', 'es', 'ly', 'ful', 'ic', 'ous', 'ism', 'ise', 'ize', 'ion', 'ment', 'en']
        # The above suffixes are removed so that some words and their suffixed word is considered as same
        for c in suffix:
            dataset[i, 1] = dataset[i, 1].replace(c, '')
            dataset[i, 2] = dataset[i, 2].replace(c, '')
        nums = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        # The numbers are replaced with a common number to ensure that there arent a lot of disntinct words merely due to the presence of numbrs
        for c in nums:
            dataset[i, 1] = dataset[i, 1].replace(c, '0')
            dataset[i, 2] = dataset[i, 2].replace(c, '0')
        subject = dataset[i, 1].split()  # the words of the subject string is obtained as a list
        body = dataset[i, 2].split()  # the words of the body string is obtained as a list
        words = subject + body
        for j in words:
            if j not in all_words:  # distinct words are recorded
                all_words[j] = b
                b += 1
        dataset_words.append(words)

    dataset_words.pop(0)  # removing the first element(empty element [] which was used to declare dataset_words)
    print('Preprocessing completed!!!')
    cnt = {}
    for j in all_words.keys():
        cnt[j] = 0.0

    for i in dataset_words:
        i = set(i)
        i = list(i)
        for j in i:
            cnt[j] += 1.0

    idf = {}  # stores the inverse document frequency of each word
    for k in cnt.keys():
        idf[k] = 1.0 + np.log((1.0 + len(dataset_words)) / (1.0 + cnt[k]))

    np.save('allwords.npy', all_words)
    np.save('IDF.npy', idf)
    print('The IDF and the dictionary of words has been saved to IDF.npy and allwords.npy')

    X = np.zeros((len(dataset_words), len(all_words)), dtype='float16')
    for i in range(len(dataset_words)):
        tf = np.zeros(len(all_words))
        l = 0
        if dataset_words[i] != []:
            for j in dataset_words[i]:
                tf[all_words[j]] += idf[
                    j]  # tf is the count of the word in the email times the idf of the word / no words in email
                # hence we directly add the idf instead adding 1 and multiplying at the end
                l += 1
            tf = tf / l  # l is the number of words in the email(normalising the term frequesncy by the number of words)
        X[i, :] = tf

    print('Encoding of the dataset completed!!!')

    y_train = y[0:10000]  # only 10000 emails are used to train
    classifier = svm.SVC(kernel='linear', C=6.0)
    print('Training in progress...')
    classifier.fit(X, y_train)
    print('Training Completed!!!')
    joblib.dump(classifier, 'trainedmodel.sav')
    print('The model has been saved to trainedmodel.sav')

def Predictor():
    saved_model = joblib.load('trainedmodel.sav')#trained model is loaded
    idf = np.load('IDF.npy', allow_pickle=True).item()
    all_words = np.load('allwords.npy', allow_pickle=True).item()

    emails = []
    directory = 'test'

    for file in sorted(os.listdir(directory), key=lambda x: int(x.replace('email', '').replace('.txt', ''))):#used to sort the emails in order
        f = open(directory + '/' + file, 'r')

        e = f.read().lower()
        emails.append(e)

    #Test emails are preprocessed
    email_words = [[]]
    for i in emails:
        characters = ['~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '[', ']', '{', '}',
                      '\\', '|', '"', "'", ':', ';', ',', '<', '.', '>', '/', '?', '/', '*', '-', '+']
        for c in characters:
            i = i.replace(c, '')
        suffix = ['ing', 'es', 'ly', 'ful', 'ic', 'ous', 'ism', 'ise', 'ize', 'ion', 'ment', 'en']
        for c in suffix:
            i = i.replace(c, '')
        nums = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        for c in nums:
            i = i.replace(c, '')

        m = i.split()
        email_words.append(m)

    email_words.pop(0)

    #Test emails are encoded into vectors
    X = np.zeros((len(email_words), len(all_words)))
    for i in range(len(email_words)):

        tf = np.zeros(len(all_words))
        l = 0
        for j in email_words[i]:
            if j in all_words:
                tf[all_words[j]] += idf[j]
                l += 1

        tf = tf / l
        X[i, :] = tf
    X_ = [[]]
    X_.append(X)
    X_.pop(0)
    prediction = saved_model.predict(X)#predictions are made
    return prediction


#Trainer() #Comment this line if only classifying the test folder is required; uncomment to re-train
predict=Predictor()
print('The classification of the test folder is:')
print(predict)