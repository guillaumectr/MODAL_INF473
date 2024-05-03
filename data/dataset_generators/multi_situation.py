from .base import DatasetGenerator
import random

def read_options(file, path="./data/dataset_generators/options/", n=-1):
    with open(path+file, 'r') as f:
        options = f.readlines()
        options = [option.strip() for option in options]
        options = [option.split("_") if len(option.split("_")) > 1 else [option, "1"] for option in options]
        options.sort(key=lambda option : float(option[-1]), reverse=True)
    return options[:n]


class MultiSituationPromptsDatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=200,
    ):
        super().__init__(generator, batch_size, output_dir)
        # self.num_images_per_label = num_images_per_label

    def create_prompts(self, labels_names):
        prompts = {}
        adjectives = read_options("adjectives.txt", n=2)
        nature = read_options("nature.txt", n=1)
        quantity = read_options("quantity.txt", n=4)
        cheese_adjectives = read_options("cheese_adjectives.txt", n=3)
        situations = read_options("situations.txt", n=8)

        for label in labels_names:
            prompts[label] = []
            for adj, n_adj in adjectives:
                n_adj = int(n_adj)
                for nat, n_nat in nature:
                    n_nat = int(n_nat)
                    for qty, n_qty in quantity:
                        n_qty = int(n_qty)
                        for cadj, n_cadj in cheese_adjectives:
                            n_cadj = int(n_cadj)
                            for sit, n_sit in situations:
                                n_sit = int(n_sit)
                                n = n_adj*n_nat*n_qty*n_cadj*n_sit//10000
                                prompts[label].append(
                                    {
                                        "prompt": f"{adj} {nat} of {qty} {cadj} {label} cheese {sit}",
                                        "num_images": n,
                                    }
                                )
        return prompts
    

    #cette fois-ci, on choisit au hasard un prompt en fonction de la pondération
    def create_prompts_random(self, labels_names):
        prompts = {}
        adjectives = read_options("adjectives.txt", n=2)
        nature = read_options("nature.txt", n=1)
        quantity = read_options("quantity.txt", n=4)
        cheese_adjectives = read_options("cheese_adjectives.txt", n=3)
        situations = read_options("situations.txt", n=8)

        adj, n_adj = zip(*adjectives)
        nat, n_nat = zip(*nature)
        qty, n_qty = zip(*quantity)
        cadj, n_cadj = zip(*cheese_adjectives)
        sit, n_sit = zip(*situations)
        adj_weigths = [int(x) for x in n_adj]
        nat_weigths = [int(x) for x in n_nat]
        qty_weigths = [int(x) for x in n_qty]
        cadj_weigths = [int(x) for x in n_cadj]
        sit_weigths = [int(x) for x in n_sit]

        for label in labels_names:
            prompts[label] = []

            for i in range(2000):
                prompts[label].append(
                                    {
                                        "prompt": f"{adj} {nat} of {qty} {cadj} {label} cheese {sit}",
                                        "num_images": 1,
                                    } 
                )
        return prompts