import pandas as pd 
from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns
from collections import defaultdict
import numpy as np 
import plotly.express as px
import random 
import statannot
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

dataset_gps = pd.read_csv('place_gps_dataset.csv') # gPlusPlaceId,Lat,Long,d1start,d1end,d2start,d2end,d3start,d3end,d4start,d4end,d5start,d5end,d6start,d6end,d7start,d7end,price
dataset_review = pd.read_csv('place_review_dataset.csv') # gPlusPlaceId,rate,reviewTime,visitTime,price
dataset_gps_review = pd.read_csv('place_gps_review_dataset.csv') # gPlusPlaceId,rate,reviewTime,visitTime,Lat,Long,d1start,d1end,d2start,d2end,d3start,d3end,d4start,d4end,d5start,d5end,d6start,d6end,d7start,d7end,price

# f1-abc: price proportion pie plot, rating pie plot, visit time pie plot.

def data_counter(dataset, dataname, datatype):
    counter_dict = defaultdict(int)
    for i in range(len(dataset)):
        key = dataset.loc[i][dataname]
        if datatype == 'int':
            key = int(key)
        counter_dict[key] += 1
    counter_dict = dict(sorted(counter_dict.items(), key=lambda item: item[1], reverse=True))
    return counter_dict

def pie_chart(dataset, dataname, datatype):
    counter_dict = data_counter(dataset, dataname, datatype)
    labels = list(counter_dict.keys())
    sizes = [counter_dict[k] for k in labels]
    size_all = np.sum(sizes)
    print(counter_dict)
    explode = [0.5 if counter_dict[k] < 10 else 0 for k in labels]
    fig1, ax1 = plt.subplots()
    wedges, texts = ax1.pie(sizes, autopct=None, shadow=False, startangle=90)
    # wedges, texts, autotexts = ax1.pie(sizes, explode = explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
    labels_legend = [str(labels[i])+':' + str(sizes[i]/size_all*100)[:4] + '%' for i in range(len(labels))]
    ax1.legend(wedges, labels_legend, title=dataname, loc="center left", bbox_to_anchor=(0.85, 0, 0.5, 1))
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

# pie_chart(dataset_gps_review, 'price', 'int')
# pie_chart(dataset_gps_review, 'rate', 'int')
# pie_chart(dataset_gps_review, 'visitTime', 'int')


# f1-cde: review time barplot count/histagram, working hour distribution histagram/fit curve, geographic distribution for all dataset (dot/color)

def reviewTime_histagram_plot(dataset):
    dataset_copy = dataset.copy()
    dataset_copy.reviewTime += 2000
    sns.histplot(data=dataset_copy, x="reviewTime", kde=True, stat = 'percent')
    plt.show()

def workhour_calculate(dataset):
    dataset_copy = dataset.copy()
    workhour_list = []
    for i in range(len(dataset_copy)):
        item = dataset_copy.loc[i]
        workhour_item = item['d1end'] - item['d1start'] + item['d2end'] - item['d2start'] + item['d3end'] - item['d3start'] + item['d4end'] - item['d4start'] + item['d5end'] - item['d5start'] + item['d6end'] - item['d6start'] + item['d7end'] - item['d7start']
        workhour_list.append(workhour_item)
    dataset_plot = pd.DataFrame(np.array(workhour_list).reshape(len(dataset_copy),1), columns=['meanhour'])
    return dataset_plot

def workhour_histagram_plot(dataset):
    dataset_copy = dataset.copy()
    dataset_plot = workhour_calculate(dataset_copy)
    sns.histplot(data=dataset_plot, x="meanhour", kde=True, stat = 'percent')
    plt.show()

def geo_distribution(dataset):
    dataset_copy = dataset.copy()
    fig = px.scatter_geo(dataset_copy, lat="Lat",lon="Long")
    fig.show()

# reviewTime_histagram_plot(dataset_gps_review)
# workhour_histagram_plot(dataset_gps_review)
# geo_distribution(dataset_gps)

# f2a: geographic location with price as dot size

def geo_distribution_with_price(dataset):
    dataset_copy = dataset.copy()
    fig = px.scatter_geo(dataset_copy, lat="Lat",lon="Long", size='price')
    fig.show()

# geo_distribution_with_price(dataset_gps)

# f2b: correlation between price (x group) and total working hour per week (boxplot)

def correlation_boxplot(dataset, dataname):
    fig, ax = plt.subplots(figsize=(5,5))
    dataset_copy = dataset.copy()
    dataset_workhour = workhour_calculate(dataset_copy)
    dataset_copy = pd.concat([dataset_copy,dataset_workhour],axis=1)
    dataset_copy.price = dataset_copy.price.astype('int')
    dataset_copy.price = dataset_copy.price.astype('string')
    ax = sns.boxplot(data=dataset_copy, x="price", y=dataname)
    pairs = [('1','2'),('1','3'),('2','3')]
    statannot.add_stat_annotation(ax,data=dataset_copy,x='price',y=dataname,box_pairs=pairs,test="t-test_ind",text_format="star",loc="outside")
    plt.subplots_adjust(top=0.7)
    plt.show()

# correlation_boxplot(dataset_gps_review, 'meanhour')

# f2c: correlation between price (x group) and average rating (boxplot)

# correlation_boxplot(dataset_gps_review, 'rate')

# f2d: correlation between price (x group) and review time (boxplot)

# correlation_boxplot(dataset_gps_review, 'reviewTime')

# f2e: correlation between price (x group) and visit times (boxplot)

# correlation_boxplot(dataset_gps_review, 'visitTime')

# tn: compare performance of different models:

def run_model(dataset, feature_list, model_name, scaler_name = None, returnFeatureImportance = False):
    dataset_copy = dataset.copy()
    dataset_copy = dataset_copy[feature_list+['price']]
    dataset_copy = dataset_copy[(dataset_copy['price']==1) | (dataset_copy['price']==2) | (dataset_copy['price']==3)]
    dataset_array = np.array(dataset_copy)

    random.seed(3)
    random.shuffle(dataset_array)

    X = [d[:-1] for d in dataset_array]
    y = [d[-1]-1 for d in dataset_array]

    X = np.float_(X)
    y = np.float_(y)

    Xtrain, Xtest = X[:(3*len(X))//4], X[(3*len(X))//4:]
    ytrain, ytest = y[:(3*len(X))//4], y[(3*len(X))//4:]

    if scaler_name != None:
        assert scaler_name in ['standard','minmax']
        scaler = StandardScaler() if scaler_name == 'standard' else MinMaxScaler()
        Xtrain = scaler.fit_transform(Xtrain)
        Xtest = scaler.fit_transform(Xtest)

    model_dict = {'base': DummyClassifier(), 'lr': LogisticRegression(), 'svm': SVC(), 'nb': GaussianNB(), 'dt': DecisionTreeClassifier(), 'rf': RandomForestClassifier(random_state=7)}
    
    clf = model_dict[model_name]
    clf.fit(Xtrain,ytrain)
    ypred = clf.predict(Xtest)

    if returnFeatureImportance == True:
        feature_importance = clf.feature_importances_ if model_name == 'rf' else clf.coef_
        return ypred, ytest, feature_importance
    else:
        return ypred, ytest
    

def compare_model(model_list, dataset, feature_list, scaler_name = None, returnFeatureImportance = False):
    for model_name in model_list:
        ypred, ytest = run_model(dataset, feature_list, model_name, scaler_name, returnFeatureImportance)
        acc = metrics.accuracy_score(ytest, ypred)
        prec = metrics.precision_score(ytest, ypred, average='weighted')
        rec = metrics.recall_score(ytest, ypred, average='weighted')
        f1 = metrics.f1_score(ytest, ypred, average='weighted')
        print(f'model: {model_name}, acc: {acc}, prec: {prec}, rec: {rec}, f1: {f1}')

model_list = ['base','lr','svm','nb','dt','rf']
feature_list_full = ['rate','reviewTime','visitTime','Lat','Long','d1start','d1end','d2start','d2end','d3start','d3end','d4start','d4end','d5start','d5end','d6start','d6end','d7start','d7end']
# compare_model(model_list,dataset_gps_review,feature_list_full,None,False)

# f3a: draw feature importance in Random Forest

def draw_feature_importance(dataset, feature_list, model_name, scaler_name):
    fig, ax = plt.subplots(figsize=(8,3))
    ypred, ytest, feature_importance = run_model(dataset, feature_list, model_name, scaler_name, True)
    plt.bar(feature_list, feature_importance, alpha = 0.5, color = 'green')
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.show()

# draw_feature_importance(dataset_gps_review, feature_list_full, 'rf', None)

# f3b: Ablation Study: All, remove each feature. Draw barplot for accuracy and F1-score.

def compare_feature(feature_delete_list_all, dataset, feature_list, model_name, scaler_name = None, returnFeatureImportance = False):
    draw_df = pd.DataFrame(columns = ['feature_delete','value','metric'])
    for feature_delete_list in feature_delete_list_all:
        feature_list_new = feature_list.copy()
        for feature_delete in feature_delete_list:
            feature_list_new.remove(feature_delete)
        ypred, ytest = run_model(dataset, feature_list_new, model_name, scaler_name, returnFeatureImportance)
        acc = metrics.accuracy_score(ytest, ypred)
        prec = metrics.precision_score(ytest, ypred, average='weighted')
        rec = metrics.recall_score(ytest, ypred, average='weighted')
        f1 = metrics.f1_score(ytest, ypred, average='weighted')
        print(f'feature_delete_list: {feature_delete_list}, acc: {acc}, prec: {prec}, rec: {rec}, f1: {f1}')
        
        if len(feature_delete_list) > 2:
            feature_delete_name = 'workhour'
        elif len(feature_delete_list) == 2:
            feature_delete_name = 'gps'
        elif len(feature_delete_list) == 0:
            feature_delete_name = 'none'
        else:
            feature_delete_name = feature_delete_list[0]
        draw_df.loc[len(draw_df.index)] = [feature_delete_name, acc, 'accuracy']
        draw_df.loc[len(draw_df.index)] = [feature_delete_name, prec, 'precision']
        draw_df.loc[len(draw_df.index)] = [feature_delete_name, rec, 'recall']
        draw_df.loc[len(draw_df.index)] = [feature_delete_name, f1, 'f1-score']
    
    fig, ax = plt.subplots()
    sns.barplot(draw_df, x = 'feature_delete', y = 'value', hue = 'metric', ax = ax, palette = 'Blues')
    
    ax.set_ylim(0.7, 0.85)
    plt.show()

# feature_delete_list_all = [[],['rate'],['reviewTime'],['visitTime'],['Lat','Long'],['d1start','d1end','d2start','d2end','d3start','d3end','d4start','d4end','d5start','d5end','d6start','d6end','d7start','d7end']]
# compare_feature(feature_delete_list_all,dataset_gps_review,feature_list_full,'rf',None,False)

# tn: compare performance of different scalers: none, standard scaler, min-max scaler

def compare_scaler(scaler_list, dataset, feature_list, model_name, returnFeatureImportance = False):
    for scaler_name in scaler_list:
        ypred, ytest = run_model(dataset, feature_list, model_name, scaler_name, returnFeatureImportance)
        acc = metrics.accuracy_score(ytest, ypred)
        prec = metrics.precision_score(ytest, ypred, average='weighted')
        rec = metrics.recall_score(ytest, ypred, average='weighted')
        f1 = metrics.f1_score(ytest, ypred, average='weighted')
        print(f'scaler: {scaler_name}, acc: {acc}, prec: {prec}, rec: {rec}, f1: {f1}')

# scaler_list = [None, 'standard', 'minmax']
# compare_scaler(scaler_list, dataset_gps_review, feature_list_full, 'rf', False)

# f4: Confusion Matrix for best classification results

def confusion_matrix_plot(dataset, feature_list, model_name, scaler_name, returnFeatureImportance):
    ypred, ytest = run_model(dataset, feature_list, model_name, scaler_name, returnFeatureImportance)
    cf_matrix = confusion_matrix(ytest, ypred)
    sns.set(font_scale=1.5)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues', xticklabels = ['\$','\$\$','\$\$\$'], yticklabels = ['\$','\$\$','\$\$\$'])
    plt.show()

confusion_matrix_plot(dataset_gps_review, feature_list_full, 'rf', None, False)


