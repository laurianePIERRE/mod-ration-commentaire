import pandas as pd
import keras
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle

def predict_comments(file,comment):
    comment=[comment]
    pickle_file= 'meta/TFIDF.pkl'
    model_file_name = "meta/SVM_model.sav"
    with open(pickle_file, 'rb') as pickle_file:
        tfidf = pickle.load(pickle_file)
    comment_tf=tfidf.transform(comment)

    loaded_model = pickle.load(open(model_file_name, 'rb'))
    print("lllllllllllll")
    predict_view= loaded_model.predict(comment_tf)
    avis= str(predict_view)
    return predict_view

def opimiseur_logisticregression (file):
    X_train, X_test, y_train, y_test = get_data_token(file)
    print ("X_train :", len(X_train.toarray()))
    print ( "X_test : ",len(X_test.toarray()))
    print("y_train : ", len(y_train.tolist()))
    print ("y_test : ", len (y_test.tolist()))
    grid = GridSearchCV(LogisticRegression(), {'C': [0.01, 0.1, 0.5, 0.75, 1, 1.5, 2, 2.25, 2.5, 3,4,5,6,7,8,9]})
    grid.fit(X_train, y_train)
    print(" best parameters", grid.best_params_)
    print(" best accuracy : ", grid.best_score_)

    df_grid = pd.DataFrame(grid.cv_results_)
    df_grid["C"] = df_grid.params.apply(lambda x: x["C"])
    print(df_grid.columns)

    # impossibilité de visualisé donc réalisation  à la main

    C_list = [ 1, 1.5, 2, 2.25, 2.5, 3,4,5,6,7,8,9]
    accuracy_list=[]
    for c in C_list:
        model = LogisticRegression(C=c, class_weight=None, dual=False, fit_intercept=True)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        model.predict(X_test)
        test_score = model.score(X_test, y_test)
        accuracy_list.append(train_score)
        print(" Pour C =", c, "l'accuracy sur le train est de : ", train_score, " et l'accuracy sur le test est de : ",
               test_score)
    plt.plot(C_list,accuracy_list)
    plt.xlabel("valeur de c")
    plt.ylabel("accuracy sur le train")
    plt.title("Recherche de l'hyperparamètre - regression logistique")
    plt.plot(6,0.9972677595628415,color="red",marker='o')
    plt.show()

def model_logisticreg(file):
    X_train, X_test, y_train, y_test = get_data_token(file)

# recherche d'hyperparamètre
    model = LogisticRegression(C=6, class_weight=None, dual=False, fit_intercept=True)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    model.predict(X_test)
    test_score = model.score(X_test, y_test)
    y_pred=model.predict(X_test)
    print(" With a Logistique regression Model: train :" + str(train_score) + ".  and test score :" + str(test_score))
    plt.figure()
    plot_curve_ROC(y_test,y_pred,"courbe ROC regression logistique")
    plt.show()
    cnf_matrix= confusion_matrix(y_test,y_pred,labels=[0,1,2,3,4])
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix_test(cnf_matrix,classes=["neutre","raciste","homophobes","sexiste","autre type d'injure"], title= "confusion matrix for logistic regression")
    plt.show()
    return model

def plot_curve_ROC(y_test,y_proba,title):
    fpr,tpr, thresholds = metrics.roc_curve(y_test,y_proba,pos_label=0)
    roc_auc = metrics.auc(fpr,tpr)
    plt.title(title)
    plt.plot(fpr,tpr,'b',label='AUC= %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def plot_confusion_matrix_test(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def randomforest_optimizeur ( file):
    X_train, X_test, y_train, y_test = get_data_token(file)
    n_estimators= [10,50,60,100,300,500,1000] # number of trees
    max_depth = [5,8,15,25,30,40,45,50,60,80,100,200]
    min_sample_split = [2,5,10,15,100]
    min_sample_leaf= [1,2,5,10]
    # hyperF = dict(n_estimators=n_estimators, max_depth=max_depth,
    #               min_samples_split=min_sample_split,
    #               min_samples_leaf=min_sample_leaf)
    #
    # gridF = GridSearchCV(RandomForestClassifier(), hyperF, cv=3, verbose=1,
    #                      n_jobs=-1)
    # gridF.fit(X_train, y_train)
    # print(" best parameters", gridF.best_params_)
    # print(" best accuracy : ", gridF.best_score_)
    list_accuracy_max=[]
    list_accuracy_estim=[]
    list_maxDepth=[]
    list_n_estimator=[]
    for es in n_estimators:
        model = RandomForestClassifier(max_depth=30, min_samples_leaf=1, min_samples_split=2,
                                                   n_estimators=es)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        list_accuracy_estim.append(train_score)
        list_n_estimator.append(es)
    for depth in max_depth :
        model = RandomForestClassifier(max_depth=depth, min_samples_leaf=1, min_samples_split=2,
                                       n_estimators=100)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        list_accuracy_max.append(train_score)
        list_maxDepth.append(depth)

    plt.plot(list_n_estimator,list_accuracy_estim)
    plt.ylabel("accuracy")
    plt.xlabel(" n_estimator")
    plt.title(" Optimisation du paramètre n_estimator")
    plt.show()

    plt.plot(list_maxDepth,list_accuracy_max)
    plt.plot(100,1, color="red", marker='o')
    plt.ylabel("accuracy")
    plt.xlabel(" max_depth")
    plt.title(" Optimisation du paramètre max_depth")
    plt.show()


def model_randomforest(file):
    X_train,X_test,y_train,y_test = get_data_token(file)

    model = RandomForestClassifier( max_depth=100, min_samples_leaf=1, min_samples_split=5, n_estimators=86)
    model.fit(X_train,y_train)
    train_score=  model.score(X_train,y_train)
    model.predict(X_test)
    test_score = model.score(X_test,y_test)
    y_pred = model.predict(X_test)

    plt.figure()
    plot_curve_ROC(y_test, y_pred, "courbe ROC random forest")
    plt.show()
    cnf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix_test(cnf_matrix,
                               classes=["neutre", "raciste", "homophobes", "sexiste", "autre type d'injure"],
                               title="confusion matrix for randomforest")
    plt.show()
    print ( " With a RandomForest Classifier Model: train :"+ str(train_score)+".  and test score :" + str(test_score))
    return model

def optimisation_svm(file):
    X_train, X_test, y_train, y_test = get_data_token(file)
    Cs=[0.001,0.01,0.1,1,10,20,30,40,50]
    gammas=[0.001,0.01,0.1,1]
    param_grid= {'C':Cs ,"gamma":gammas}
    grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid , cv=3)
    grid.fit(X_train,y_train)
    print(" best parameters", grid.best_params_)
    print(" best accuracy : ", grid.best_score_)
    accuracy_Cs=[]
    accuracy_gammas=[]
    for c in Cs:
        model=svm.SVC(C=c, gamma=0.1)
        model.fit(X_train,y_train)
        accuracy_Cs.append(model.score(X_train,y_train))
    for gamma in gammas:
        model = svm.SVC(C=10, gamma=gamma)
        model.fit(X_train,y_train)
        accuracy_gammas.append(model.score(X_train,y_train))

    plt.plot(Cs,accuracy_Cs)
    plt.plot(10, 1, color="red", marker='o')
    plt.xlabel("valeurs de C")
    plt.ylabel("valeurs de l'accuracy")
    plt.title("optimisation du paramètre C")
    plt.show()

    plt.plot(gammas, accuracy_gammas)
    plt.plot(0.1, 1, color="red", marker='o')
    plt.xlabel ("valeurs de gamma")
    plt.ylabel ("valeurs de l'accuracy")
    plt.title ("optimisation du paramètre gamma")
    plt.show()

def model_svm (file):
    X_train, X_test, y_train, y_test = get_data_token(file)
    model = svm.SVC(C=10, gamma=0.1)
    model.fit(X_train, y_train)
    file_name="meta/SVM_model.sav"
    pickle.dump(model,open(file_name,'wb'))
    print(" model bien enregistrer")
    train_score = model.score(X_train, y_train)
    model.predict(X_test)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    #
    # plt.figure()
    # plot_curve_ROC(y_test, y_pred, "courbe ROC svm")
    # plt.show()
    # cnf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
    # np.set_printoptions(precision=2)
    # plt.figure()
    # plot_confusion_matrix_test(cnf_matrix,
    #                            classes=["neutre", "raciste", "homophobes", "sexiste", "autre type d'injure"],
    #                            title="confusion matrix for svm")
    # plt.show()
    print(" With an SVM : train :" + str(train_score) + ".  and test score :" + str(test_score))
    return model


def optimiseur_knn (file):
    X_train, X_test, y_train, y_test = get_data_token(file)
    grid_params= {'n_neighbors': [3,5,11,15,19], 'weights': ['uniform','distance'], 'metric' : ['euclidean','manhattan']}
    grid= GridSearchCV(KNeighborsClassifier(),grid_params,verbose=1, cv=3, n_jobs=1)
    grid.fit(X_train, y_train)
    print(" best parameters", grid.best_params_)
    print(" best accuracy : ", grid.best_score_)
    n_neighbors= [3,5,11,15,19]
    weights= ['uniform', 'distance']
    metric= ['euclidean','manhattan']
    accuracy_neighbors=[]
    accuracy_weight=[]
    accuracy_metrics=[]
    for n in n_neighbors:
        model=KNeighborsClassifier(n_neighbors=n,weights='distance')
        model.fit(X_train,y_train)
        accuracy_neighbors.append(model.score(X_train,y_train))
    for w in weights:
        model=KNeighborsClassifier(n_neighbors=3, weights=w)
        model.fit(X_train,y_train)
        accuracy_weight.append(model.score(X_train,y_train))
    for m in metric:
        model = KNeighborsClassifier(n_neighbors=3, weights='distance',metric=m)
        model.fit(X_train, y_train)
        accuracy_metrics.append(model.score(X_train, y_train))

    plt.plot(n_neighbors,accuracy_neighbors)
    plt.plot(3, 1, color="red", marker='o')
    plt.xlabel("valeurs de n_neighbors")
    plt.ylabel("valeurs de l'accuracy")
    plt.title("optimisation du paramètre n_neighbors")
    plt.show()

    plt.bar(weights,accuracy_weight)
    plt.xticks(weights, ("uniform", "distance"))
    plt.ylabel("valeurs de l'accuracy")
    plt.title("optimisation du paramètre weight")
    plt.show()

    plt.bar(metric, accuracy_metrics)
    plt.xticks(metric, ("euclidean", "manhattan"))
    plt.ylabel("valeurs de l'accuracy")
    plt.title("optimisation du paramètre metric")
    plt.show()

def model_knn (file):
    X_train, X_test, y_train, y_test = get_data_token(file)
    model = KNeighborsClassifier( n_neighbors=3, weights="distance",metric="euclidean")
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    model.predict(X_test)
    test_score = model.score(X_test, y_test)
    print(" With an knn : train :" + str(train_score) + ".  and test score :" + str(test_score))
    return model


def get_data_token(file):
    dt = pd.read_csv(file, error_bad_lines=False, sep='\t')
    X = dt['comments']
    Y = dt[' label']
    X=put_comments_in_str(X)
    pickle_name='meta/TFIDF.pkl'
    X_tfidf=tokenizer_data(X,pickle_name)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, Y,
                                                        test_size=0.3,
                                                        random_state=0)

    return X_train,X_test,y_train,y_test


def put_comments_in_str(data):
    """ avoid some type bugs putting each comments in string type  """
    data_in_str = []
    for comment in data :
        data_in_str.append(str(comment))
    return data_in_str

def tokenizer_data (data,pickle_name):
   vectorizer =  TfidfVectorizer()
   tfidf_model =  vectorizer.fit(data)
   pickle.dump(tfidf_model, open(pickle_name,'wb'))
   x_tfidf=vectorizer.transform(data)
   return  x_tfidf
