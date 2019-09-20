from bdd import *
import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split

def put_data_ingured( file, sep , liens,name_article,name_client,theme_client):
    comments = pd.read_csv (file, encoding = "ISO-8859-1", sep=sep)
    comms=comments["commentaire"]
    labels=comments["label"]
    put_client(name_client,theme_client)
    id_client=get_id_client(name_client,theme_client)
    ajout_article(liens,name_article,id_client)
    id_article = get_id_article(liens,name_article)


    for com,label in zip(comms,labels) :
        ajout_com_insultant(com, label, name_client, id_article)


