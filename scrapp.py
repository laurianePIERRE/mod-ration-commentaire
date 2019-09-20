


from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
import requests
import re


def scrapp_accueil(url):

    r = requests.get(url)
    print(r)
    raw = r.text
    list_lien = []
    list_theme = []

    soup = bs(raw, 'html.parser')

    for balise in soup.findAll("a", {'class':'first-lien'}):

        lien = balise["href"]

        list_lien.append("http://www.jeuxvideo.com"+ str(lien))

    for element in list_lien:

        theme=re.sub("www.jeuxvideo.com/[A-Za-z]*/[A-Za-z\-\]*/[0-9]*/","",element)
        theme=re.sub(".htm","",theme)
        list_theme.append(theme)

    return list_lien, list_theme

def create_list_comm(url):

    l, auteur = import_com(url)

    comments = []
    for com, aut in zip(l, auteur):
        data = {
            "commentaire" : com ,
            "auteur" : aut
        }
        comments.append(data)


    return comments

def import_com(url):

   # print(url)
    raw = urlopen(url).read().decode('latin-1', 'ignore')
    soup = bs(raw, 'html.parser', from_encoding='latin-1')
    all_title_balise = soup.findAll('div',{'class':'txt-msg text-enrichi-forum'})
    lien=[]
    for i, b in enumerate(all_title_balise):
       # print("balise", i, b.findChild("p").text)
        lien.append(b.findChild("p").text)
    liste_auteur=[]
    for i in range(len(all_title_balise)):
        auteur=soup.findAll('div', {'class':'bloc-header'})[i].span.span.text
        liste_auteur.append(auteur)
   # print(len(lien), len(liste_auteur))
    return (lien,liste_auteur)
