"""importazione librerie"""
import gensim
import gensim.corpora as corpora
import matplotlib.colors as mcolors
import pyLDAvis.gensim_models
from gensim.models.nmf import Nmf
from matplotlib import pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from wordcloud import WordCloud

# Abilita la visualizzazione nel IPython Notebook.
pyLDAvis.enable_notebook()


class TopicModeling:
    """ Classe che riceve i testi dei contratti puliti negli script precedenti tramite
        file csv ed esegue una Topic Modeling sia attraverso LDA che NMF con rappresentazioni
        grafiche. """

    def __init__(self, csv_path, n_components=None, learning_decay=None, max_iter=None):
        """ :Params csv_path: path del file csv contenente i documenti ripuliti
            :param n_components: range di topics da testare nella cross validation
            :param learning_decay: range di pesi relativi al tasso di degrado dell' apprendimento da testare nella cross
                                   validation
            :param max_iter: range di iterazioni del modello da testare nella cross validation"""

        self.n_components, self.learning_decay, self.max_iter = n_components, learning_decay, max_iter

        # Importazione documenti
        self.documents = []
        with open(csv_path, "r", encoding="utf8") as file:
            for row in file:
                self.documents.append(row)
        file.close()

        data_words = [doc.split() for doc in self.documents]

        self.id2word = corpora.Dictionary(data_words)
        # Creazione della bag of words
        self.corpus = [self.id2word.doc2bow(text) for text in data_words]

        self.tf, self.tf_feature_names, self.tf_vector = self.count_vec()

        self.best_values = self.cross_validation(self.n_components, self.learning_decay, self.max_iter)

    def count_vec(self):
        """ Questa funzione ci permette di prende del testo e trasformarlo in un vettore contenente
            il numero di occorrenze dei tokens.

            :return: term_freq: matrice contenente le frequenze delle parole per ogni documento
            :return: term_freq_features_names: nomi delle features
            :return: make_vector: risultato del CountVectorizer """

        make_vector = CountVectorizer(max_df=0.30,  # usato per eliminare i termini troppo frequenti
                                      min_df=0.05,  # usato per eliminare i termini troppo poco frequenti
                                      stop_words='english')  # lingua in base alla quale escludere le stopwords

        term_freq = make_vector.fit_transform(self.documents)
        term_freq_features_names = make_vector.get_feature_names()

        return term_freq, term_freq_features_names, make_vector

    def cross_validation(self, n_components=None, learning_decay=None, max_iter=None):
        """ Questa funzione ci permette di attuare un processo di selezione dei valori ottimali
            per la Topic Modelling.
            Nel caso in cui l' utente non specifichi i parametri richiesti abbiamo discrezionalmente
            scelto come range di numero di topic quello compreso tra 5 e 20, per il learning decay i valori
            0.5, 0.7 e 0.9 e per il numero massimo di iterazioni un range da 1 a 10..

            :param n_components: numero topic da testare
            :param learning_decay: valori di decay da testare
            :param max_iter: numero di iterazioni da testare
            :return: i migliori parametri ottenuti dalla cross validation """

        if n_components is None:
            n_components = [n_topic for n_topic in range(5, 20)]  # numero topic da testare
        if learning_decay is None:
            learning_decay = [.5, .75, .9]  # valori di decay da testare
        if max_iter is None:
            max_iter = [iterations for iterations in range(1, 10)]  # numero di iterazioni da testare

        # Con Nmf non funziona purtroppo
        lda = LatentDirichletAllocation()

        # Parametri da testare
        search_params = {'n_components': n_components,
                         'learning_decay': learning_decay,
                         'max_iter': max_iter}

        # Creazione Grid-Search
        model = GridSearchCV(lda, param_grid=search_params)
        model.fit(self.tf)

        best_model = model.best_estimator_

        # Log Likelihood Score
        print("Log Likelihood Score: ", model.best_score_)  # più è alto meglio performa il modello

        # Perplexity
        print("Model Perplexity: ", best_model.perplexity(self.tf), "\n")  # più è basso meglio performa il modello

        # print(model.best_params_)  # check best params

        return model.best_params_

    def lda_model(self):
        """ Creazione modello di Latent Dirichilet Allocation.

            I parametri: **num_topics**, **decay** e **iterations** vengono decisi tramite
            la cross validation effettuata dalla funzione *cross_validation*, in base ai valori
            passati alla classe stessa

            :return: grafici online intetrattivi stivati nella cartella chiamata link_grafici e
                     word-cloud per ogni topic contenenti le parole più rilevanti """

        lda = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                              id2word=self.id2word,
                                              num_topics=self.best_values['n_components'],
                                              decay=self.best_values['learning_decay'],
                                              iterations=self.best_values['max_iter'],
                                              random_state=100,
                                              update_every=1,
                                              chunksize=100,
                                              passes=10,
                                              alpha="auto")

        # Print risultati
        print("RISULTATI LDA"), self.results(lda)

        # Creazione grafici interattivi
        site = pyLDAvis.gensim_models.prepare(lda, self.corpus, lda.id2word)
        pyLDAvis.save_html(site, "link_grafici/LDATopicModeling.html")

        return self.make_graphs(lda)

    def nmf_model(self):
        """ Creazione modello di Non-Negative Matrix Factorization.

            Il parametro: **num_topics** è lo stesso estratto per la LDA tramite la funzione
            *cross_validation*, in base ai valori passati alla classe stessa
            :return: word-cloud per ogni topic contenenti le parole più rilevanti """

        nmf = Nmf(corpus=self.corpus,
                  id2word=self.id2word,
                  num_topics=self.best_values['n_components'],
                  random_state=100,
                  chunksize=100,
                  passes=10)

        # Print risultati
        print("RISULTATI NMF"), self.results(nmf)

        return self.make_graphs(nmf)

    def make_graphs(self, model):
        """ Genera delle word-cloud per il modello che gli viene passato come parametro

            :param model: modello di LDA o NMF """

        # Palette di colori
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

        # Creazione layout word-cloud
        cloud = WordCloud(background_color='#f4ece6',
                          width=2500,
                          height=1800,
                          max_words=5,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[topic_number],
                          min_font_size=3,
                          prefer_horizontal=1.0)

        # Identificazione topics del modello
        topics = model.show_topics(formatted=False)

        """ Configurazione numero di colonne del grafico in base al numero dei topics """
        n_col = 1
        if self.best_values['n_components'] % 2 == 0:
            n_col = 2

        fig, axes = plt.subplots(self.best_values['n_components'] // n_col, n_col, figsize=(10, 10))

        """ Costruzione grafico """
        for topic_number, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)  # aggiunta asse
            topic_words = dict(topics[topic_number][1])  # dizionario del topic con relativi valori per ogni sua parola
            cloud.generate_from_frequencies(topic_words, max_font_size=300)  # word-cloud del topic
            plt.gca().imshow(cloud)
            plt.gca().set_title(f'Topic{str(topic_number + 1)}', fontdict=dict(size=16))  # titolo topic
            plt.gca().axis('off')

        # Ulteriori settaggi del grafico finale
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()

    def results(self, model):
        """ Visualizzazione risultati topic
        :param model: modello utilizzato per la topic-modeling
        :return: print di topic identificati con relative parole e valori di queste ultime """

        topics = model.show_topics(formatted=False)
        for t in topics:
            print(f'Topic: {t[0] + 1}')
            for word_list in t[1][:5]:
                print(word_list[0], f': {word_list[1]}')
            print("\n")


if __name__ == "__main__":
    test = TopicModeling("Deposito_contratti/cleaned_dataset_lemma.csv",
                         n_components=[4, 6, 8, 10],  # best = 10
                         learning_decay=[.75, .9],  # best = .75
                         max_iter=[3, 4, 5])  # best = 5

    test.lda_model()
    test.nmf_model()
