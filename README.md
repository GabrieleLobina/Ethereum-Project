# README #

## PROGETTO LOBINA-CHESSA ##

### SMART CONTRACT WITH ETHERIUM ###

L' obiettivo di questo progetto è quello di effettuare una topic modeling sui testi 
degli smart contract di ethereum in modo da comprendere in quali ambiti essi vengono utilizzati principalmente.   
I contratti in questione dovranno essere estratti dal sito Etherscan.io, ripuliti e poi utilizzati in due 
modelli di topic modeling, LDA (Latent Dirichlet Allocation) e NMF (Non-Negative Matrix Factorization).


#### L' ordine di esecuzione dei programmi è il seguente: ####

- Branch Web-scraping-e-Parsing: estrazione dei contratti dalla piattaforma Etherscan.io
- Branch stopwords-cleaning: pulizia dei testi dei contratti e *Topic modeling*


#### Nella cartella *Deposito_contratti* del branch stopwords-cleaning saranno proposti tre file: #####
* contracts.csv (contiene circa 5000 contratti)
* cleaned_dataset_lemma.csv (pulizia + lemmatization di contracts.csv)
* cleaned_dataset_stem.csv (pulizia + stemming di contracts.csv)

Questi propongono i risultati dei file ***scraping.py*** e ***Topic_modeling.py***


