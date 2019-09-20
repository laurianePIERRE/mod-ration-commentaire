# coding: utf8
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
import urllib
import requests
import re
from bdd import *
from IA_function import *
import pickle
import pandas as pd
from IA_function import *
#url = "https://bruiter.com/actu"
def get_article_url_bruter():
    """  return a tuple contains the list of url and the list of the name of the articles in the site in the category actu """
    url = "https://bruiter.com/actu"
    r=requests.get(url)
    compteur = 0
    raw=r.text
    soup=bs(raw,'html.parser')
    url_list=[]
    name_list=[]
    for balise in soup.findAll("a",{'class':'jsx-2122937052 topic-title'}) :
        if compteur<= 2 :
            compteur+=1
            lien = balise["href"]
            name= re.sub("^/topic/[0-9\-]*","", lien)
            name = re.sub("\-", " ", name)
            name_list.append(name)
            url_list.append("https://bruiter.com"+str(lien))
    return url_list,name_list

def get_comment_through_url_bruter():
    comments_list=[]
    compteur=0
    url_list, list_name_article = get_article_url_bruter()
    for url, name_article in zip(url_list, list_name_article):
        comments_by_article=[]
        #r_article = requests.get(url)
        print("url ",url)
        headers = {
            'User-Agent': 'My User Agent 1.0',
            'From': 'joetheboy@gmail.com'
        }
       #  request = urllib.request(url,)
       # raw= urllib.urlopen(request).read()
       #  reque = urlopen(request).read().decode('latin-1', 'ignore')
        raw= requests.get(url, headers=headers)
        print ("raw : ", raw)
        soup = bs(raw.text, 'html.parser', from_encoding='latin-1')
        print ("ok")
        # raw = r_article.text.decode('latin-1','ignore')
        # #raw = urlopen(url).read().decode('latin-1')
        # soup = bs(raw,'html.parser', from_encoding='latin-1',)
        all_title_balise = soup.findAll('div', {'class': 'jsx-227213379 message-text'})
        print(" wait ---------------  scrapping of"+url+" is loading ")
        for i, b in enumerate(all_title_balise):
           #if compteur <=7:
            #    compteur += 1
            if ( b.findChild("p") == None):  #  sinon bug
                continue
            else :
                com= b.findChild("p").string
                comments_by_article.append(str(com))


        comments_list=comments_list+comments_by_article
    return comments_list


def put_bruiter_comments_in_database ():
    comment_list=get_comment_through_url_bruter()
    comment_list=verify_list_com(comment_list)
    url_article,  list_name  = get_article_url_bruter()
    put_client("bruiter", "commentaire pour model")
    id_client = get_id_client("bruiter", "commentaire pour model")
    for url, name, in zip( url_article,list_name,) :
        ajout_article(url, name, id_client)
        id_article = get_id_article(url, name)
        for comm in comment_list:
            ajout_com ( comm,"utilisateurBruiter",id_article)


def verify_list_com(list_com):
    new_list_com = []
    for com in list_com :
        for i in range(len(list_com)):
            if com != list_com[i] :
                new_list_com.append(com)
    return new_list_com


def get_data_comment_bruiter():
    # recover comments from the database into a list of lists
    cnx,cursor=get_connection()
    get_data="SELECT * FROM commentaire where  auteur=%s;"
    cursor.execute(get_data, ["utilisateurBruiter"])
    data = cursor.fetchall()
    cursor.close()
    return data



def data_in_dictionnary():
    ## if the comment has been labelised , put it on a dict to transform in a data frame
    data = get_data_comment_bruiter()
    list_of_dict=[]
    for ele in data :
        d = {
            "id" : ele[0],
            "comments" : ele[1],
            " label" : ele[2],

            }
        if ele[2] is None:
            break;

        list_of_dict.append(d)
    return list_of_dict


def data_in_csv(file_name):
    liste=data_in_dictionnary()
    df = pd.DataFrame.from_dict(liste)
    df.to_csv(file_name,sep='\t')



def get_comments_avis_bruiter():
    dt = pd.read_csv("bruiter.csv", error_bad_lines=False, sep='\t')
    comment_list = dt['comments']
    label_list = dt[' label']
    id_list = dt ["id"]
    return  id_list, comment_list, label_list



## insérer commentaire à partir du site
def put_bruiter_comments_in_database ():
    comment_list=get_comment_through_url_bruter()
    url_article,  list_name  = get_article_url_bruter()
    put_client("bruiter", "commentaire pour model")
    id_client = get_id_client("bruiter", "commentaire pour model")
    for url, name, in zip( url_article,list_name,) :
        ajout_article(url, name, id_client)
        id_article = get_id_article(url, name)
        for comm in comment_list:
            if comm!="None":
                ajout_com ( str(comm), "utilisateurBruiter",str(id_article))



def predict_bruiter_comments (comment) :
    comment= [comment]
    file ="meta/comments_in_database.csv"
    pickle_file = 'meta/TFIDF.pkl'
    model = model_randomforest(file)
    with open(pickle_file, 'rb') as pickle_file:
        tfidf = pickle.load(pickle_file)
    comment_tf = tfidf.transform(comment)
    predict_view = model.predict(comment_tf)
    # print(" predict_view :" + str(predict_view) + " type : " + str(type(predict_view)))
    avis = str(predict_view)
    # print(" avis :" + avis + " type : " + str(type(avis)))
    # if avis == "[0]":
    #     print("le commentaire " + str(comment[0]) + " est à modérer")
    # if avis == "[1]":
    #     print("le commentaire " + str(comment[0]) + " est bon ")

    return predict_view


