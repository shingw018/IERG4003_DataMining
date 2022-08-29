import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn import preprocessing


def readtrain():
    col_names = ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
                 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
    csv = pd.read_csv("train.csv", header=None, names=col_names, skiprows=[0])

    response = csv['Exited']
    csv = csv.drop(columns=['Exited'])
    csv = csv.drop(columns=['RowNumber'])
    csv = csv.drop(columns=['CustomerId'])
    csv = csv.drop(columns=['Surname'])

    csv = pd.DataFrame(pd.get_dummies(csv))

    for i in range(len(csv['Balance'])):
        if((csv['Balance'][i] == 0) and (csv['Tenure'][i] > 8)):
            csv['Balance'][i] = csv['Balance'].mean() + \
                csv['Balance'].std()*1
        if((csv['Balance'][i] == 0) and (csv['Tenure'][i] > 5)):
            csv['Balance'][i] = csv['Balance'].mean()

    csv['Balance'] = preprocessing.minmax_scale(
        csv[['Balance']], feature_range=(0, 1))

    return(csv, response)


def readtest():
    col_names = ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
                 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    csv = pd.read_csv("assignment-test.csv", header=None,
                      names=col_names, skiprows=[0])

    _id = csv['RowNumber']
    csv = csv.drop(columns=['RowNumber'])
    csv = csv.drop(columns=['CustomerId'])
    csv = csv.drop(columns=['Surname'])

    csv = pd.DataFrame(pd.get_dummies(csv))

    for i in range(len(csv['Balance'])):
        if((csv['Balance'][i] == 0) and (csv['Tenure'][i] > 8)):
            csv['Balance'][i] = csv['Balance'].mean() + \
                csv['Balance'].std()*1
        if((csv['Balance'][i] == 0) and (csv['Tenure'][i] > 5)):
            csv['Balance'][i] = csv['Balance'].mean()

    csv['Balance'] = preprocessing.minmax_scale(
        csv[['Balance']], feature_range=(0, 1))

    return (csv, _id)


def treeclass(trainset, testset, response, id):
    bayes = CategoricalNB(alpha=0, fit_prior=True, class_prior=None)

    bayes.fit(trainset, response)
    pred = bayes.predict(testset)
    pred_response = []
    _id = []
    for i in range(len(pred)):
        pred_response.append(pred[i])
        _id.append(id[i])
    _dict = {"id": _id, "Response": pred_response}
    _format = pd.DataFrame(_dict, columns=['id', 'Response'])
    _format.to_csv('submission_2_Bayes.csv', index=False, header=True)


trainset, response = readtrain()
testset, id = readtest()

treeclass(trainset, testset, response, id)
