import numpy as np
import pandas as pd


workclass = {'Private':0,'Self-emp-not-inc':1, 'Self-emp-inc':2, 'Federal-gov':3,
             'Local-gov':4, 'State-gov':5, 'Without-pay':6, 'Never-worked':7}
education = {'Bachelors':0,'Some-college':1, '11th':2, 'HS-grad':3,
             'Prof-school':4, 'Assoc-acdm':5, 'Assoc-voc':6, '9th':7,
             '7th-8th':8,'12th':9, 'Masters':10, '1st-4th':11, '10th':12,
             'Doctorate':13, '5th-6th':14, 'Preschool':15}
marital_status = {'Married-civ-spouse':0, 'Divorced':1, 'Never-married':2,
                  'Separated':3, 'Widowed':4, 'Married-spouse-absent':5,
                  'Married-AF-spouse':6}
occupation = {'Tech-support':0, 'Craft-repair':1, 'Other-service':2, 'Sales':3,
              'Exec-managerial':4, 'Prof-specialty':5, 'Handlers-cleaners':6,
              'Machine-op-inspct':7, 'Adm-clerical':8, 'Farming-fishing':9,
              'Transport-moving':10, 'Priv-house-serv':11, 'Protective-serv':12,
              'Armed-Forces':13}
relationship = {'Wife':0, 'Own-child':1, 'Husband':2, 'Not-in-family':3,
                'Other-relative':4, 'Unmarried':5}
race = {'White':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':2, 'Other':3, 'Black':4}
sex = {'Female':0, 'Male':1}
native_country = {'United-States':0, 'Cambodia':1, 'England':2, 'Puerto-Rico':3, 'Canada':4,
                  'Germany':5, 'Outlying-US(Guam-USVI-etc)':6, 'India':7, 'Japan':8,
                  'Greece':9, 'South':10, 'China':11, 'Cuba':12, 'Iran':13, 'Honduras':14,
                  'Philippines':15, 'Italy':16, 'Poland':17, 'Jamaica':18, 'Vietnam':19,
                  'Mexico':20, 'Portugal':21, 'Ireland':22, 'France':23, 'Dominican-Republic':24,
                  'Laos':25, 'Ecuador':26, 'Taiwan':27, 'Haiti':28, 'Columbia':29, 'Hungary':30,
                  'Guatemala':31, 'Nicaragua':32, 'Scotland':33, 'Thailand':34, 'Yugoslavia':35,
                  'El-Salvador':36, 'Trinadad&Tobago':37, 'Peru':38, 'Hong':39, 'Holand-Netherlands':40}
target = {'>50K':0,'<=50K':1}

labelname = {0:'age',1:workclass,2:'fnlwgt',3:education,
             4:'education-num',5:marital_status,6:occupation,
             7:relationship,8:race,9:sex,10:'capital-gain',
             11:'capital-loss',12:'hours-per-week',13:native_country,14:target}
categorical_list = [1,3,5,6,7,8,9,13,14]
continous_list = [0,2,4,10,11,12]

filename = 'adult.data'


def read_data(f_name)->list:
    data = []
    with open(f_name,'r') as f:
        file = f.readlines()

        for index, line in enumerate(file):
            line = line.strip().split(', ')
            linelist = []
            for idx, e in enumerate(line):
                if e == '?':
                    linelist.append(-1)
                elif idx not in categorical_list:
                    try:
                        linelist.append(int(e))
                    except ValueError:
                        print(index, idx,line)
                else:
                    linelist.append(labelname[idx][e])
            if len(linelist)!=0:
                data.append(linelist)
    return data


def preprocess(data):

    data = np.array(data)
    #n_record, n_col = data.shape

    # discretization equal frequency bucketing
    for i in continous_list:
        ftr = data[:,i]
        result = pd.qcut(ftr[ftr != -1], 5, duplicates='drop')
        count = pd.qcut(ftr[ftr != -1], 5, duplicates='drop').value_counts().index
        #print(count)
        count_dict = {}
        for idx, item in enumerate(count):
            count_dict[item] = idx
        #print(count)
        new_ftr= [count_dict[i] for i in result]
        #print(new_ftr)


        ftr[ftr != -1]=np.array(new_ftr)
        data[:, i] = ftr

    return data








