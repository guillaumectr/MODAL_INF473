from .base import DatasetGenerator


class MultiSituationPromptsDatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=200,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label

    def create_prompts(self, labels_names):
        prompts = {}
        with open("./data/dataset_generators/options/adjectives.txt", 'r') as f:
            adjectives = f.readlines()
            adjectives = [label.strip() for label in adjectives]
        with open("./data/dataset_generators/options/nature.txt", 'r') as f:
            nature = f.readlines()
            nature = [label.strip() for label in nature]
        with open("./data/dataset_generators/options/quantity.txt", 'r') as f:
            quantity = f.readlines()
            quantity = [label.strip() for label in quantity]
        with open("./data/dataset_generators/options/cheese_adjectives.txt", 'r') as f:
            cheese_adjectives = f.readlines()
            cheese_adjectives = [label.strip() for label in cheese_adjectives]
        with open("./data/dataset_generators/options/situations.txt", 'r') as f:
            situations = f.readlines()
            situations = [label.strip() for label in situations]

        for label in labels_names:
            prompts[label] = []
            for adj in adjectives[:5]:
                for nat in nature[:5]:
                    for qty in quantity[:5]:
                        for cadj in cheese_adjectives[:5]:
                            for sit in situations[:5]:
                                prompts[label].append(
                                    {
                                        "prompt": f"{adj} {nat} of {qty} {cadj} {label} cheese {sit}",
                                        "num_images": self.num_images_per_label,
                                    }
                                )
        return prompts
