import sys
import pandas
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB


def gen_train_test_data():
    test_ratio = .3 
    df = pandas.read_csv('./yelp labelled.txt', sep='\t', header=None, encoding='utf-8')
    count_vect = CountVectorizer()
    X = count_vect.fit_transform(df[0])

    y = df[1].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=0)

                
    return X_train, X_test, y_train, y_test


def multinomial_nb(X_train, X_test, y_train, y_test):
    # TODO: fill this function
    # train by X_train and y_train
    # report the predicting accuracy for both the training and the test data 
    tmp_rs = X_train.shape
    #計算train data中，有幾筆是positive，有幾筆是negative
    y_len = len(y_train)
    c_positive = 0
    c_negative = 0
    for i in range(0,y_len):
        if y_train[i] == 1:
            c_positive = c_positive + 1
        else:
            c_negative = c_negative + 1

    ##Prior
    prior_pos = c_positive / len(y_train)
    prior_neg = c_negative / len(y_train)

    ##Likelihood
    #把positive和negtive之下，各自有出現字的次數加總起來
    tmp_X = X_train.toarray()
    col_pos = []
    col_neg = []
    for i in range(0,len(y_train)):
        if y_train[i] == 1:
            col_pos.append(tmp_X[i])
        else:
            col_neg.append(tmp_X[i])
    #將list 轉成array
    Arr_col_pos = np.array(col_pos)
    Arr_col_neg = np.array(col_neg)
    #加總各個單詞，個別在pos和neg出現的次數
    sum_pos_Ni = Arr_col_pos.sum(axis=0)
    sum_neg_Ni = Arr_col_neg.sum(axis=0)
    #把pos和neg中，各個單詞出現的次數全部加總起來
    sum_neg_N = 0
    sum_pos_N = 0
    for i in range(0,len(sum_pos_Ni)):
        sum_pos_N = sum_pos_N + sum_pos_Ni[i]

    for i in range(0,len(sum_neg_Ni)):
        sum_neg_N = sum_neg_N + sum_neg_Ni[i]
    #計算likelihood機率
    prob_likeli_pos = []
    prob_likeli_neg = []
    tmp = 0
    for i in range(0,tmp_rs[1]):
        tmp = (sum_pos_Ni[i] + 1) / (sum_pos_N + tmp_rs[1]) 
        prob_likeli_pos.append(tmp)
        tmp = (sum_neg_Ni[i] + 1) / (sum_neg_N + tmp_rs[1])
        prob_likeli_neg.append(tmp)

    ##帶入train data 去計算準確率
    X_train_arr = X_train.toarray()
    post_prob_pos = []
    for i in range(0,len(y_train)):
        tmp_post = 1
        for j in range(0,len(X_train_arr[i])):
            tmp_post = tmp_post * ((prob_likeli_pos[j]) ** (X_train_arr[i][j]))

        t = prior_pos * tmp_post
        post_prob_pos.append(t)

    post_prob_neg = []
    for i in range(0,len(y_train)):
        tmp_post = 1
        for j in range(0,len(X_train_arr[i])):
            tmp_post = tmp_post * ((prob_likeli_neg[j]) ** (X_train_arr[i][j]))

        t = prior_neg * tmp_post
        post_prob_neg.append(t)     

    #比較哪個後驗機率比較大，去判別屬於哪一個class
    pridect_result = []
    for i in range(0,len(y_train)):
        if post_prob_pos[i] > post_prob_neg[i]:
            y = 1
        else:
            y = 0
        
        pridect_result.append(y)

    #計算準確率
    cnt = 0
    for i in range(0,len(y_train)):
        if pridect_result[i] == y_train[i]:
            cnt = cnt +1
    print("following multinomial distribution, accuracy of train data :", (cnt/tmp_rs[0])) 

    ##帶入test data 去做測試 X_test
    #計算positive的後驗機率，並且储存在一個List
    X_test_arr = X_test.toarray()
    post_prob_pos = []
    for i in range(0,len(y_test)):
        tmp_post = 1
        for j in range(0,len(X_test_arr[i])):
            tmp_post = tmp_post * ((prob_likeli_pos[j]) ** (X_test_arr[i][j]))

        t = prior_pos * tmp_post
        post_prob_pos.append(t)  

    post_prob_neg = []
    for i in range(0,len(y_test)):
        tmp_post = 1
        for j in range(0,len(X_test_arr[i])):
            tmp_post = tmp_post * ((prob_likeli_neg[j]) ** (X_test_arr[i][j]))

        t = prior_neg * tmp_post
        post_prob_neg.append(t)   

    #比較哪個後驗機率比較大，去判別屬於哪一個class
    pridect_result = []
    for i in range(0,len(y_test)):
        if post_prob_pos[i] > post_prob_neg[i]:
            y = 1
        else:
            y = 0
        
        pridect_result.append(y)

    #計算準確率
    cnt = float(0)
    for i in range(0,len(y_test)):
        if pridect_result[i] == y_test[i]:
            cnt = cnt +1.0
    
    print("following multinomial distribution, accuracy of test data :", (cnt/len(y_test))) 



def bernoulli_nb(X_train, X_test, y_train, y_test):
    # TODO: fill this function
    # train by X_train and y_train
    # report the predicting accuracy for both the training and the test data
    tmp_rs = X_train.shape
    #計算train data中，有幾筆是positive，有幾筆是negative
    y_len = len(y_train)
    c_positive = 0
    c_negative = 0
    for i in range(0,y_len):
        if y_train[i] == 1:
            c_positive = c_positive + 1
        else:
            c_negative = c_negative + 1

    ##Prior
    prior_pos = c_positive / len(y_train)

    prior_neg = c_negative / len(y_train)

    ##Likelihood
    #把positive和negtive之下，各自有出現字的次數加總起來
    tmp_X = X_train.toarray()

    #把所有出現次數大於1的全部轉成 1 ，因為Bernoulli Distribution只判斷他有或沒有出現
    for i in range(0,tmp_rs[0]):
        for j in range(0,tmp_rs[1]):
            if tmp_X[i][j] > 1:
                tmp_X[i][j] = 1

    #把屬於postive類的DATA放在col_pos，屬於negtive類的DATA放在col_neg，這是存成List，所以下面要轉成array
    col_pos = []
    col_neg = []
    for i in range(0,len(y_train)):
        if y_train[i] == 1:
            col_pos.append(tmp_X[i])
        else:
            col_neg.append(tmp_X[i])

    #將list 轉成 array
    Arr_col_pos = np.array(col_pos)
    Arr_col_neg = np.array(col_neg)                            

    #加總各個單詞，個別在pos和neg出現的次數
    sum_pos_Ni = Arr_col_pos.sum(axis=0)
    sum_neg_Ni = Arr_col_neg.sum(axis=0)

    #計算likelihood機率
    prob_likeli_pos = []
    prob_likeli_neg = []
    tmp = 0
    for i in range(0,tmp_rs[1]):
        tmp = (sum_pos_Ni[i] + 1) / (c_positive + tmp_rs[1]) 
        prob_likeli_pos.append(tmp)
        tmp = (sum_neg_Ni[i] + 1) / (c_negative + tmp_rs[1]) 
        prob_likeli_neg.append(tmp)

    ##帶入train data 去計算準確率
    X_train_arr = X_train.toarray()
    post_prob_pos = []
    for i in range(0,len(y_train)):
        tmp_post = 1
        for j in range(0,len(X_train_arr[i])):
            tmp_post = tmp_post * ((prob_likeli_pos[j]) ** (X_train_arr[i][j]))

        t = prior_pos * tmp_post
        post_prob_pos.append(t)

    post_prob_neg = []
    for i in range(0,len(y_train)):
        tmp_post = 1
        for j in range(0,len(X_train_arr[i])):
            tmp_post = tmp_post * ((prob_likeli_neg[j]) ** (X_train_arr[i][j]))

        t = prior_neg * tmp_post
        post_prob_neg.append(t)     

    #比較哪個後驗機率比較大，去判別屬於哪一個class
    pridect_result = []
    for i in range(0,len(y_train)):
        if post_prob_pos[i] > post_prob_neg[i]:
            y = 1
        else:
            y = 0
        
        pridect_result.append(y)

    #計算準確率
    cnt = 0
    for i in range(0,len(y_train)):
        if pridect_result[i] == y_train[i]:
            cnt = cnt +1
    print("following bernoulli distribution, accuracy of train data :", (cnt/tmp_rs[0]))

    ##帶入test data 去做測試 X_test
    #計算positive的後驗機率，並且储存在一個List
    X_test_arr = X_test.toarray()
    post_prob_pos = []
    for i in range(0,len(y_test)):
        tmp_post = 1
        for j in range(0,len(X_test_arr[i])):
            tmp_post = tmp_post * ((prob_likeli_pos[j]) ** (X_test_arr[i][j]))

        t = prior_pos * tmp_post
        post_prob_pos.append(t) 

    post_prob_neg = []
    for i in range(0,len(y_test)):
        tmp_post = 1
        for j in range(0,len(X_test_arr[i])):
            tmp_post = tmp_post * ((prob_likeli_neg[j]) ** (X_test_arr[i][j]))

        t = prior_neg * tmp_post
        post_prob_neg.append(t)   

    #比較哪個後驗機率比較大，去判別屬於哪一個class
    pridect_result = []
    for i in range(0,len(y_test)):
        if post_prob_pos[i] > post_prob_neg[i]:
            y = 1
        else:
            y = 0
        
        pridect_result.append(y)

    cnt = 0
    for i in range(0,len(y_test)):
        if pridect_result[i] == y_test[i]:
            cnt = cnt +1
    
    print("following bernoulli distribution, accuracy of test data :", cnt/len(y_test)) 
      
    


def main(argv):
    X_train, X_test, y_train, y_test = gen_train_test_data()

    multinomial_nb(X_train, X_test, y_train, y_test)
    bernoulli_nb(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main(sys.argv)


