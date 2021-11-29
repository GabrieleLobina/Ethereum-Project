# Web scraping #

### Obiettivi: ###

* Estrazione dei link contenenti i codici dei contratti dalla pagina https://etherscan.io/contractsVerified
* Ingresso nei suddetti link ed estrazione del codice che compone i contratti
* Inserimento del codice in un file csv

### Passi preliminari ###

* Scaricare le varie librerie presenti nel file *requirements.txt*
* Per effettuare l' accesso alle varie pagine web verrà utilizzato ***selenium***, il quale necessita di determinati 
  driver.  
      Il programma è predisposto per l' utilizzo o di Chrome oppure di Safari:  
  - Per Chrome = **ChromeDriver** (ottenibile al seguente link:
    https://chromedriver.chromium.org/downloads)
  - Per Safari = non necessita di alcun driver, se non del settaggio delle impostazioni del browser descritto al seguente
    link:
    https://www.browserstack.com/guide/run-selenium-tests-on-safari-using-safaridriver
    
* Per utilizzare Safari: commentare il codice alle righe 20 e 63 poi decommentare alle righe 21 e 64

#### N.B.: Accertarsi sempre di avere il driver compatibile con la versione del proprio browser 

### Descrizione programma ###

Scraping.py contiene due classi: **Page** e **ObtainContracts**.

La prima ci permette di entrare nelle pagine iniziali ed estrarre i link nei quali poi entreremo per copiare i codici
dei contratti.  
La seconda entra nei suddetti link, copia il codice dei contratti e lo inserisce in un file csv stivato nella cartella
Deposito_contratti.

Sito sul quale verrà eseguito lo scraping:
https://etherscan.io/contractsVerified  
Questo ci permette di estrarre solo gli ultimi 500 contratti stipulati con Ethereum perciò, il file scraping.py verrà
eseguito più volte fino all' ottenimento di un numero di campioni sufficienti per eseguire un' analisi di topic
modeling.


