from dataset_generators.multi_situation import MultiSituationPromptsDatasetGenerator

with open("./list_of_cheese.txt", "r") as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]

dg = MultiSituationPromptsDatasetGenerator(None)
print(dg.create_prompts(labels))