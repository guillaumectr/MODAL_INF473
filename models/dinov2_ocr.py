import torch
import torch.nn as nn
import easyocr
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from unidecode import unidecode
import numpy as np

from models.dinov2 import DinoV2Finetune

cheese_list = [["BRIE", "MELUN"], ["CAMEMBERT", "PRESIDENT"], ["EPOISSES", "BERTHAUT"], ["FOURME", "AMBERT"], ["RACLETTE"], ["MORBIER"], ["NECTAIRE"],
                ["POULIGNY", "PIERRE"], ["ROQUEFORT", "SOCIETE","ROQUEBELLE"], ["COMTE"], [], ["PECORINO"], ["NEUFCHATEL"], ["CHEDDAR"],
                ["BUCHETTE", "BUCHE"], ["PARMESAN", "PARMIGIANO"], ["FELICIEN", "SAINT-FELICIEN"], ["MONT", "OR"], ["STILTON", "STILLTON"], ["SCARMOZA"], ["CABECOU", "ROCAMADOUR"],
                ["BEAUFORT"], ["MUNSTER"], ["CHABICHOU", "POITOU"], ["TOMME"], ["REBLOCHON"], ["EMMENTAL"], ["FETA"], ["OSSAU", "IRATY"], ["MIMOLETTE"],
                ["MAROILLES"], ["GRUYERE"], ["MOTHAIS", "FEUILLE"], ["VACHERIN", "FRIBOURGEOIS"], ["MOZZARELLA"], ["TETE", "MOINES", "BELLELAY", "TETEDEMOINE"], ["FRAIS"]]

def tensor_to_numpy(tensor):
    numpy_image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return numpy_image

class DinoV2OCR(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.dino = DinoV2Finetune(num_classes, frozen, unfreeze_last_layer)
        self.reader = easyocr.Reader(['fr'])

    def tag_identifier(self, image):
        numpy_image = tensor_to_numpy(image)
        result = self.reader.readtext((numpy_image * 255/numpy_image.max()).astype(np.uint8))
        img_words = [unidecode(result[i][1]).upper() for i in range(len(result))]
        #list of words identified in the image in capital letters ans without the accents

        if len(img_words) > 0:
            for label in range(len(cheese_list)):
                A = False
                for word in cheese_list[label]:
                    for img_word in img_words:
                        if label == 20 or label == 9: #pb de similarité entre cabecou et chabichou : on exige une correspondance + forte pour le 21ème fromage. Aussi problème avec le comté
                            A = A or (fuzz.ratio(word, img_word) > 80)
                        else:
                            A = A or (fuzz.ratio(word, img_word) > 70)
                if A:
                    z = torch.zeros(len(cheese_list))
                    z[label] = 1
                    return z
        return None

    def forward(self, x):
        y = [self.tag_identifier(x_i) for x_i in x]
        x = self.dino(x)

        for i in range(len(x)):
            if y[i] is not None:
                x[i] = y[i]

        return x
