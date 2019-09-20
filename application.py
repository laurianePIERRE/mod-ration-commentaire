# coding: utf8
from flask import render_template, request, Flask
from bdd import *
from IA_function import *
from scrappBuiter import *
from get_data import *
from manual_comments import *


APP = Flask("nom")


# un évenement = une route + méthods

@APP.route("/", methods=['GET','POST'])
def main():
  return render_template("Accueil.html",**locals())



@APP.route("/GetDataset", methods=[ 'GET','POST'])
def GetDataset():
    tenth_first("http://www.jeuxvideo.com",'JeuxVideo.com','videogames ')
    return render_template("database_completed.html", **locals())


@APP.route("/GetCommentsIngured", methods=[ 'GET','POST'])
def GetCommentsIngured():
    put_data_ingured("commentaire_manuelle/BFM_Facebook.txt", '\t',
                     "https://www.facebook.com/search/top/?q=bfm&epa=SEARCH_BOX", "BFM_FACEBOOK", "manuelle",
                     "reseaux sociaux")
    put_data_ingured("commentaire_manuelle/French.csv", ';',
                     "https://data.world/wordlists/dirty-naughty-obscene-and-otherwise-bad-words-in-french",
                     "dataset words", "dataword", " network datasets")
    put_data_ingured("commentaire_manuelle/classification.txt","\t","http://www.topito.com/top-propos-homophobes-scandaleux-encore-beaucoup-travail-faire-envie-de-vomir","ajout_injures","manuelle"," internet")
    put_data_ingured ( "commentaire_manuelle/data_promo.csv",";","no liens","promo","manuelle","promo")
    return render_template("database_completed.html", **locals())


@APP.route("/createfich", methods=[ 'GET','POST'])
def createfich():
    name_fic="meta/comments_in_database.txt"
    data = get_data_comment()
    data_in_csv(data,name_fic)
    model_svm(name_fic)

    return render_template("fichierCree.html",**locals())

@APP.route('/negvspos', methods=['POST'])
def negvspos():

    commentaire = recuperer()
    if request.method == 'POST' :
        for id_com in request.form.items():
            print (id_com)
            label = id_com[1]
            labelise(id_com[0],label)
            commentaire = recuperer()
        return render_template('labelisation_jvc.html', **locals())


@APP.route("/bruitter_completed", methods = ["GET","POST"])
def bruitter_completed():
    put_bruiter_comments_in_database()
    put_data_ingured("commentaire_manuelle/data_promo_test.csv", ";", "no liens", "promo", "utilisateurBruiter", "promo")
    comment_list = recover_bruiter_comment()
    for id, com, avis, id_article, auteur in comment_list:
        avis_pred = predict_comments("comments_in_database.csv", com)
        labelise(id, str(avis_pred[0]))

    return render_template("bruitter_completed.html")

@APP.route ("/predict_page", methods = ["GET","POST"])
def predict_page():
    comment_list = recover_bruiter_comment()

    if request.method == 'POST':

            for form in request.form.items():

                id_com = form[0]
                label =form[1]
                labelise_spe(label,id_com)
                print(" ok modifier")

            comments_list_predict=recover_bruiter_comment()
    return render_template("prediction.html", **locals())

@APP.route("/bruitter_in_db", methods= ["GET","POST"])
def bruitter_in_db():

    comment_list = recover_bruiter_comment()
    print( " hello")
    for id, com, avis, id_article, auteur in comment_list:
        changed_author(id, "anciens bruitter")
        print(auteur, "  anciens ")


    return render_template("new_database.html")

@APP.route("/drop", methods =['GET',"POST"])
def drop():
    reinitialize_data_base()
    return render_template("data_base_empty.html")

if __name__ == "__main__":
    APP.run(debug=True)

