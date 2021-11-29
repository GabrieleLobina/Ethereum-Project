""" Importazione delle librerie """
from urllib.parse import urlparse, urljoin
import pyperclip
from bs4 import BeautifulSoup as Bs
from selenium import webdriver


class Page:
    """ Entra nella pagina e ricerca dei particolari link.

        Parametri:
            * **address**: pagina dalla quale vogliamo estrarre i links"""

    def __init__(self, address):
        self.address = address
        self.links = {}
        self.url = urlparse(self.address)

        try:
            driver = webdriver.Chrome('Drivers/Chrome/chromedriver.exe')  # se si vuole usare Chrome
            # driver = webdriver.Safari()  # se si vuole usare Safari
            driver.get(self.address)
            html_source = driver.page_source
            driver.quit()
            soup = Bs(html_source, "html.parser")
            self.links = {a['href'] for a in soup.find_all("a", href=True, class_="hash-tag text-truncate")}
            self.parse_urls()
        except Exception:
            pass

    def parse_urls(self):
        """ Completamento links + #code """

        external_links = []
        for link in self.links:
            external = (urlparse(link))
            external = urlparse(urljoin(self.url.geturl(), external.path + "#code"))
            external_links.append(external)
        self.links = {link.geturl() for link in external_links}


class ObtainContracts:
    """ Estrae i codici dei contratti dai link presenti nelle pagine che gli passiamo.

        Parametri:
            * **page_links**: lista contenente i link delle pagine i cui sono presenti contratti"""

    def __init__(self, page_links):
        self.page_links = page_links

        for page in self.page_links:
            self.save_results(page)

    def scraping(self, url):
        """ Individuazione e copia del codice del contratto. La funzione utilizzerà l' url passatogli
        come parametro per aprire una pagina di Chrome e poi cliccherà nel bottone che permette
        di copiare il codice del contratto per poi ritornarlo. Nel caso in cui il programma
        riscontri degli errori la funzione terminerà avvisandoci.

         :param url: url del sito dal quale estrarre il codice del contratto
         :return: testo del contratto copiato """

        driver = webdriver.Chrome('Drivers/Chrome/chromedriver.exe')  # se si vuole usare Chrome
        # driver = webdriver.Safari()  # se si vuole usare Safari
        try:
            driver.get(url)
            driver.execute_script("window.scrollTo(0, 500)")  # scroll-down della pagina per raggiungere il bottone
            driver.find_element_by_xpath(
                '/html/body/div[1]/main/div[4]/div[3]/div[2]/div/div[8]/div[2]/div[2]/div[1]/div[2]/span/a[1]').click()
            # print(pyperclip.paste())  # Test
            driver.quit()
            return pyperclip.paste()

        except Exception:
            print("Error: text not copied")
            driver.quit()
            pass

    def save_results(self, page):
        """ Salva i risultati in un file csv che andrà riposto dentro la cartella Deposito_contratti.
        Se questo non esiste verrà creato, se invece esiste verrà semplicemente aggiornato.

         :param page: pagina contenente i link in cui sono presenti i contratti  """

        contracts_already_extracted = set()  # contratti già estratti in passato
        try:
            # Se il file esiste viene aggiornato
            print("Aggiornamento file 'contracts.csv' presente nella cartella 'Deposito_contratti'")

            with open("Deposito_contratti/contracts.csv", "r", encoding="utf-8") as controller:
                """ Lettura del file finalizzata all' esclusione dei doppioni"""

                text = ""
                for row in controller:
                    for letter in row:
                        if letter == "þ":
                            contracts_already_extracted.add(text)
                            text = ""
                        else:
                            text += letter
            controller.close()

            with open("Deposito_contratti/contracts.csv", "a", encoding="utf-8") as file:
                """ Aggiornamento file"""
                site = Page(page)
                for link in site.links:
                    text = self.scraping(link)
                    if type(text) == str:
                        if text not in contracts_already_extracted:
                            file.write(text + "þ")  # "þ" è il delimitatore di riga
                            print("Text copied")
                        else:
                            print("Contratto già presente nel file")
                            # se lo scraping è andato a buon fine e il
                            # testo non è già stato estratto in passato,
                            # allora quest' ultimo viene aggiunto al file
            file.close()

        except Exception:
            # Se il file non esiste viene creato
            with open("Deposito_contratti/contracts.csv", "w", encoding="utf-8") as file:
                print("Creazione file 'contracts.csv' nella cartella 'Deposito contratti'")

                site = Page(page)
                for link in site.links:
                    text = self.scraping(link)
                    if type(text) == str:
                        file.write(text + "þ")
                        print("Text copied")
            file.close()


if __name__ == "__main__":
    extraction = ObtainContracts(["https://etherscan.io/contractsVerified/1?ps=100",
                                  "https://etherscan.io/contractsVerified/2?ps=100",
                                  "https://etherscan.io/contractsVerified/3?ps=100",
                                  "https://etherscan.io/contractsVerified/4?ps=100",
                                  "https://etherscan.io/contractsVerified/5?ps=100"])
