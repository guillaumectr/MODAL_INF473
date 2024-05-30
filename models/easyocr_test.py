import torch
import easyocr
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from unidecode import unidecode
import os

#We list the words of each class that we want to identify. We have a problem with chèvre and buchette de chèvre and sèvres and other cheese made of chèvre (ex chabichou), so we do not try to identify chèvre
cheese_list = [["BRIE", "MELUN"], ["CAMEMBERT", "PRESIDENT"], ["EPOISSES", "BERTHAUT", "L'EPOISSES"], ["FOURME", "AMBERT"], ["RACLETTE"], ["MORBIER"], ["NECTAIRE"],
                ["POULIGNY", "PIERRE"], ["ROQUEFORT", "SOCIETE","ROQUEBELLE"], ["COMTE"], [], ["PECORINO"], ["NEUFCHATEL"], ["CHEDDAR"],
                ["BUCHETTE", "BUCHE"], ["PARMESAN", "PARMIGIANO"], ["FELICIEN", "SAINT-FELICIEN"], ["MONT", "OR"], ["STILTON", "STILLTON"], ["SCARMOZA"], ["CABECOU", "ROCAMADOUR"],
                ["BEAUFORT"], ["MUNSTER"], ["CHABICHOU", "POITOU"], ["TOMME"], ["REBLOCHON"], ["EMMENTAL"], ["FETA", "GRECQUE", "GREEK", "GREC", "GRECE"], ["OSSAU", "IRATY"], ["MIMOLETTE"],
                ["MAROILLES"], ["GRUYERE"], ["MOTHAIS", "FEUILLE"], ["VACHERIN", "FRIBOURGEOIS"], ["MOZZARELLA"], ["TETE", "MOINES", "BELLELAY", "TETEDEMOINE"], ["FRAIS"]]
               

reader =  easyocr.Reader(['fr'])

#This function returns (True,  cheese_label) if OCR has detected something, or (False, "") if nothing significant is detected
def tag_identifier(path):
    result = reader.readtext(path)
    img_words = []
    for i in range(len(result)):
        img_words += unidecode(result[i][1]).upper().split()
    #list of words identified in the image in capital letters ans without the accents

    if len(img_words) > 0:
        for label in range(len(cheese_list)):
            A = False
            for word in cheese_list[label]:
                for img_word in img_words:
                    # print(word)
                    # print(img_word)
                    # print(fuzz.ratio(word, img_word))
                    if label == 20 or label == 9: #pb de similarité entre cabecou et chabichou : on exige une correspondance + forte pour le 21ème fromage. Aussi problème avec le comté
                        A = A or (fuzz.ratio(word, img_word) > 80)
                    else:
                        A = A or (fuzz.ratio(word, img_word) > 75)
            if A:
                return True, label + 1
    return False, ""

folder_path = "C:/Users/nicol/OneDrive/Documents/Polytechnique/2A/P3/Modal/MODAL_INF473/dataset/val/ MONT D'OR"

i = 0
for path in [folder_path + "/" + x for x in os.listdir(folder_path)]:
    print(i)
    print(tag_identifier(path))
    i += 1


# tag_identifier("C:/Users/nicol/OneDrive/Documents/Polytechnique/2A/P3/Modal/MODAL_INF473/dataset/val/CHEDDAR/000022.jpg")
print(fuzz.ratio("fromage", "fourme"))