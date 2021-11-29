from Data_Cleaning import DatasetPulito
from Topic_modeling import TopicModeling

if __name__ == "__main__":
    # Pulizia contratti
    contratti = DatasetPulito("Deposito_contratti/contracts.csv")

    # Topic modeling su cleaned_dataset_lemma.csv
    topic_lemma = TopicModeling("Deposito_contratti/cleaned_dataset_lemma.csv",
                                n_components=[4, 6, 8, 10],
                                learning_decay=[.75, .9],
                                max_iter=[3, 4, 5])
    topic_lemma.lda_model()
    topic_lemma.nmf_model()

    # Topic modeling su cleaned_dataset_stem.csv
    topic_stem = TopicModeling("Deposito_contratti/cleaned_dataset_stem.csv",
                               n_components=[4, 6, 8, 10],
                               learning_decay=[.75, .9],
                               max_iter=[3, 4, 5])
    topic_stem.lda_model()
    topic_stem.nmf_model()
