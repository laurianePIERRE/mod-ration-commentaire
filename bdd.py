
# coding: utf8
import mysql.connector

from scrapp import *


def get_connection():
    cnx = mysql.connector.connect( user='root', password = 'root',database='commentaire')
    cursor = cnx.cursor()
    return cnx, cursor

def tenth_first(url,client,theme):
    put_client(client, theme)
    links, name_article = scrapp_accueil(url)

    for link, name_article in zip(links, name_article):
        print('processing link {} and theme {}'.format(link, theme))
        print('...')
        remplir_base(link, name_article,client,theme)

def put_client(name,theme):
    cnx, cursor = get_connection()
    add_client = ("INSERT INTO client "
               "(Name, theme)"
               "VALUES (%s,%s)")

    cursor.execute(add_client, (name,theme))
    cnx.commit()
    cursor.close()
    cnx.close()

def remplir_base(url, name_article,client,theme):
    #    functio who fill the SQL database with the customer, article of the web site and the comments
    id_client=get_id_client(client,theme)
    ajout_article(url,name_article,id_client)
    id_article = get_id_article(url,name_article)
    commentaires = create_list_comm(url)
    for comment in commentaires:
        ajout_com(comment['commentaire'], comment['auteur'],id_article)

def get_id_article(url, name_article):
    cnx, cursor = get_connection()
    select_id = " SELECT idarticle  from article WHERE url_article=%s AND name_article=%s;"
    cursor.execute(select_id, (url, name_article))
    data = cursor.fetchall()
    cursor.close()
    result = data[0][0]
    return result

def get_id_client(name,theme):
    cnx, cursor = get_connection()
    select_id = " SELECT idClient from client WHERE Name=%s AND theme=%s;"
    cursor.execute(select_id, (name, theme))
    data = cursor.fetchall()
    cursor.close()
    result = data[0][0]
    return result

def get_label(id_com):
    cnx, cursor = get_connection()
    auteur ="utilisateurBruiter"
    Select_label = "SELECT label from commentaire where id_commentaire=%s AND auteur=%s ;"
    cursor.execute(Select_label,(id_com,auteur))
    data = cursor.fetchall()
    result = data[0][0]
    cursor.close()
    return result

def ajout_com(comment, auteur, article):
    cnx, cursor = get_connection()
    add_com= ("""INSERT INTO commentaire 
              (commentaire,label, auteur, article)
              VALUES (%s,%s,%s,%s)""")
    cursor.execute(add_com, (comment,"0", auteur, article))
    cnx.commit()
    cursor.close()
    cnx.close()

def ajout_com_insultant(comment, label, auteur, article):
    cnx, cursor = get_connection()
    add_com= ("""INSERT INTO commentaire 
              (commentaire,label, auteur,article)
              VALUES (%s,%s,%s,%s)""")
    cursor.execute(add_com, (comment, label, auteur, article))
    cnx.commit()
    cursor.close()
    cnx.close()

def ajout_article(adresse, name_article,client):
    cnx, cursor = get_connection()
    add_article= ("INSERT INTO article "
                  "(name_article, url_article,id_client) "
                  "VALUES (%s,%s,%s)")
    cursor.execute(add_article, ( name_article, adresse,client))
    cnx.commit()
    cursor.close()
    cnx.close()


def add_warring() :
    cnx, cursor = get_connection()
    add_warring =(" INSERT INTO  warring "
                 "(id_warring, name) VALUES (%s,%s)")
    cursor.execute(add_warring, ("0"," aucun"))
    cursor.execute(add_warring,("1", " racisme"))
    cursor.execute(add_warring, ( "2", "homophobe"))
    cursor.execute(add_warring, ( "3", "haine"))
    cursor.execute(add_warring, ("4","sexisme"))
    cursor.execute(add_warring, ("5"," autre"))
    cnx.commit()
    cursor.close()
    cnx.close()

def labelise(idcommentaire,label):
    cnx, cursor = get_connection()
    insert_avis = "UPDATE commentaire.commentaire SET label= %s WHERE id_commentaire = %s;"
    cursor.execute(insert_avis, (label, idcommentaire))
    cnx.commit()
    cursor.close()

def labelise_spe( avis ,idcommentaire) :
    cnx, cursor = get_connection()
    insert_avis = ("UPDATE commentaire.commentaire SET label= %s WHERE id_commentaire = %s;")
    cursor.execute(insert_avis, (avis, idcommentaire))
    cnx.commit()
    cursor.close()

def recuperer():
    cnx,cursor= get_connection()
    select_id = " SELECT * from commentaire WHERE auteur != 'utilisateurBruiter' AND auteur != 'dataset'"
    cursor.execute(select_id)
    data = cursor.fetchall()
    cnx.close()
    return data

def changed_author(id, auteur):
    cnx, cursor = get_connection()
    insert_avis = ("UPDATE commentaire.commentaire SET auteur= %s WHERE id_commentaire = %s;")
    cursor.execute(insert_avis, (auteur, id))
    cnx.commit()
    cursor.close()

def recover_bruiter_comment() :
    """ recover only the comments which the client is bruiter"""
    cnx, cursor = get_connection()
    select_client = (
        " select * from commentaire.commentaire where commentaire.auteur=%s")
    cursor.execute(select_client, ["utilisateurBruiter"])
    list_comments_bruiter = cursor.fetchall()

    return list_comments_bruiter

def verify_comment_client_no_bruiter(com,id_article):
    """ verity if client's comments isn't bruiter"""
    cnx,cursor = get_connection()
    select_client = " Select id_commentaire From commentaire.client" \
                    " join commentaire.article join commentaire.commentaire" \
                    "where client.name = %s;"
    cursor.execute(select_client,["bruiter"])
    list_comments_bruiter = cursor.fetchall()
    id_com= get_id_commment(com,id_article)
    if id_com  in list_comments_bruiter:
        return False
    else:
        return True

def get_id_commment (com, id_article):
    cnx, cursor = get_connection()
    select_id = " SELECT id_commentaire from commentaire WHERE commentaire=%s AND id_article=%s;"
    cursor.execute(select_id, (com, id_article))
    data = cursor.fetchall()
    cursor.close()
    result = data[0][0]
    return result

def reinitialize_data_base():
    cnx, cursor = get_connection()
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0 ;")
    cursor.execute("TRUNCATE TABLE commentaire ;")
    cursor.execute(" TRUNCATE  TABLE client ;")
    cursor.execute("TRUNCATE TABLE article ;")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1 ;")


    cursor.execute("ALTER TABLE client AUTO_INCREMENT = 1 ;")
    cnx.commit()
    cursor.close()

def verify_new_artiche(article_test):
    list_articles = get_all_article()
    for article in list_articles:
        if article== article_test :
            return False
        else:
            return True


def get_all_article():
    cnx, cursor = get_connection()
    get_article =" SELECT name_article from article"
    cursor.execute(get_article)
    list_all_articles=cursor.fetchall()
    cnx.commit()
    cursor.close()
    return list_all_articles


