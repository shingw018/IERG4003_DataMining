import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing


def readtrain():
    col_names = ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
                 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Response']
    csv = pd.read_csv("insurance-train.csv", header=None,
                      names=col_names, skiprows=[i for i in range(0, 300001)])

    le = preprocessing.LabelEncoder()
    le.fit(csv['Gender'])
    csv['Gender'] = le.transform(csv['Gender'])
    le.fit(csv['Vehicle_Age'])
    csv['Vehicle_Age'] = le.transform(csv['Vehicle_Age'])
    le.fit(csv['Vehicle_Damage'])
    csv['Vehicle_Damage'] = le.transform(csv['Vehicle_Damage'])

    response = csv['Response']
    csv = csv.drop(columns=['Response'])
    return(csv, response)


def readtest():
    col_names = ['id', 'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
                 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
    csv = pd.read_csv("insurance-test.csv", header=None,
                      names=col_names, skiprows=[0])

    le = preprocessing.LabelEncoder()
    le.fit(csv['Gender'])
    csv['Gender'] = le.transform(csv['Gender'])
    le.fit(csv['Vehicle_Age'])
    csv['Vehicle_Age'] = le.transform(csv['Vehicle_Age'])
    le.fit(csv['Vehicle_Damage'])
    csv['Vehicle_Damage'] = le.transform(csv['Vehicle_Damage'])

    _id = csv['id']
    csv = csv.drop(columns=['id'])
    return (csv, _id)


def treeclass(trainset, testset, response, id):
    svm = SVC(C=5, kernel='linear', random_state=0)

    svm.fit(trainset[['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
                      'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']], response)

    pred = svm.predict(testset)
    pred_response = []
    _id = []
    for i in range(len(pred)):
        pred_response.append(pred[i])
        _id.append(id[i])
    _dict = {"id": _id, "Response": pred_response}
    _format = pd.DataFrame(_dict, columns=['id', 'Response'])
    _format.to_csv('submission_1_SVM.csv', index=False, header=True)


trainset, response = readtrain()
testset, id = readtest()

treeclass(trainset, testset, response, id)
