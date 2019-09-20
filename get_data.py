import mysql.connector
import pandas as pd
from IA_function import *
def get_connection():
    cnx = mysql.connector.connect( user='root', password = 'root',database='commentaire')
    cursor = cnx.cursor()
    return cnx, cursor


def get_data_comment():
    # recover comments from the database into a list of lists
    cnx,cursor=get_connection()
    get_data="SELECT * FROM commentaire"
    cursor.execute(get_data)
    data = cursor.fetchall()
    cursor.close()
    return data


def data_in_dictionnary(data):
    ## if the comment has been labelised , put it on a dict to transform in a data frame
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


def data_in_csv(data,file_name):
    liste=data_in_dictionnary(data)
    df = pd.DataFrame.from_dict(liste)
    df.to_csv(file_name,sep='\t')

# data=get_data_comment()
# data_in_csv(data,"com0205.csv")
#
# model_logisticreg("com0205.csv")
# model_randomforest("com0205.csv")



