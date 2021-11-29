# README #

In questo branch si attua la pulizia dei testi dei contratti estratti in
Web-Scraping-e-Parsing e successivamente li si utilizza in 2 modelli di *topic modeling*.

>Per ottenere il file csv (aggiornato) generato nel branch Web-Scraping-e-Parsing è sufficiente scrivere nel terminale:  
"git checkout origin/Web-Scraping-e-Parsing Deposito_contratti/contracts.csv"

### Obiettivi ###

* *Pulizia contratti*
  * eliminazione punteggiatura 
  * eliminazione stop-words
  * eliminazione sintassi Solidity
  * eliminazione parole prive di significato 
  * *Lemmatization* / *Stemming*

* *Topic modelling* 
    * applicazione LDA
    * applicazione NMF
    * esposizione risultati

L' esecuzione del programma, nel caso in cui non avvenisse tramite main.py, deve seguire il seguente ordine:
prima Data_Cleaning.py e poi Topic_modeling.py

### Data_Cleaning.py ###

Questo file è composto dalla sola classe **DatasetPulito**

Tramite questa apriamo il file *contracts.csv*, creato nel branch Web-Scraping-e-Parsing e inseriamo i contratti in 
un set.  
Attraverso le regex gestiamo la punteggiatura ed estraiamo le parole dai CamelCase. Distinguiamo la pulizia  
per la *lemmatization* da quella per lo *stemming* ma in ogni caso, per entrambi i procedimenti escludiamo  
le parole ritenute obsolete ai fini della futura ***Topic modeling***.  
La classe genera due file csv, uno comprendente la pulizia con *lemmatization* e l' altro con *stemming*,   
entrambi salvati all' interno della cartella Deposito_contratti.  
I risultati possono essere osservati anche attraverso due dataset presenti negli attributi: ***lemma_dataset*** e ***stem_dataset***


### Topic_modeling.py ###
Questo file è composto dalla sola classe **TopicModeling**  

Estraiamo i contratti appena ripuliti dal file che abbiamo selezionato (cleaned_dataset_lemma.csv oppure 
cleaned_dataset_stem.csv) e salviamo i contratti in una lista.  
Generiamo un *CountVectorizer* il quale ci servirà per attuare una cross-validation sul modello di LDA.  
Identifichiamo i migliori parametri da passare ai modelli riguardanti numero di topics e poi, per
la LDA anche i valori ottimali di *learning_decay* e *max_iter*.
Tramite gensim generiamo una LDA e una NMF e stampiamo i risultati sia su terminale, tramite
una rappresentazione delle prime 5 parole per importanza per topic, e a video tramite delle word-cloud.
Per quanto riguarda il primo modello, il programma genera una rappresentazione grafica interattiva relativa ai topic
individuati osservabile al sito salvato nella cartella *link_grafici*.
