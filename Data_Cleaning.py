""" Importazione librerie """
import re

import nltk
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet as wn


class DatasetPulito:
    """ Classe che estrae i contratti da un file csv e restituisce una lista e un
        dataset contenente gli stessi puliti. Si può scegliere tra una pulizia che comprende
        un processo di stemming o uno di lemmatizzazione. """

    # Lista di parole da escludere dai testi
    bad_words = ["import", "zero", "update",
                 "true", "false", "value",
                 "tokens", "token", "target",
                 "set", "return", "address",
                 "string", "account", "user",
                 "owner", "add", "sender",
                 "spender", "recipient", "bytes",  # bytes: tenere/lasciare
                 "transfer", "memory", "liquidity",  # transfer, memory, liquidity: tenere/lasciare
                 "pool", "virtual", "indexed",  # pool, virtual: tenere/lasciare
                 "err", "callable", "requirements",  # requirements: tenere/lasciare
                 "error", "allowance", "allowances",
                 "emit", "emits", "vault",
                 "id", "ids", "decimals",
                 "map", "integers", "number",
                 "log", "buffer", "overflow",
                 "amounts", "notice", "unsigned",
                 "remaining", "approve", "revert",
                 "ratio", "underlying", "path",
                 "approved", "ignored", "shares",
                 "use", "depositor", "key",
                 "length", "index", "want", "deadline",
                 "index", "balance", "strategy", "fee",
                 "position", "length", "contract",
                 "read", "caller", "revert", "operator",
                 "interface", "initialize", "module",
                 "ether", "pay", "word", "contain", "remove",
                 "revert", "decode", "price", "mm", "available",
                 "ignore", "approval", "constant", "controller",
                 "withdraw", "storage", "inner", "exist",
                 "delete", "type", "collect", "setups",
                 "setup", "identifier", "sign", "role",
                 "roles", "grant", "pause", "self",
                 "total", "approval", "remain", "stop",
                 "color", "price", "position", "fee",
                 "file", "order", "indicate", "request",
                 "proposal", "gain", "counterpart",
                 "success", "rate", "math", "locker",
                 "asset", "payload", "send", "encode",
                 "lower", "upper", "growth", "exclude",
                 "min", "extension", "max", "creator",
                 "xi", "current", "factor", "active",
                 "https", "protocol", "second", "random",
                 "extension", "bots", "change", "base",
                 "enable", "mean", "calculate", "initial",
                 "point", "tx", "count", "start",
                 "fail", "enable", "subtract", "exact",
                 "array", "implementation", "list",
                 "version", "like", "non", "enable",
                 "line", "burn", "reason", "units",  # decidere se tenere o togliere "burn"
                 "claim", "signature", "11", "receive",
                 "support", "access", "context", "allow",
                 "make", "claim", "safe", "sol", "factory",
                 "addition", "interfaces", "hash", "receiver",
                 "router", "payable", "pair", "reserve",
                 "time", "fix", "control", "minter",
                 "deposit", "operations", "assembly",
                 "increase", "decrease", "root", "limit",
                 "hook", "deposit", "check", "open", "push",
                 "previous", "reflect", "note", "12", "18",
                 "underline", "actual", "title",
                 "selector", "sub", "low", "disable",
                 "execute", "pack", "latest", "super",
                 "distribute", "holders", "delegate",
                 "old", "invalid", "create", "source",
                 "permission", "previous", "minimum",
                 "division", "open", "need", "title",
                 "trigger", "cost", "assert", "round",
                 "modify", "single", "raise", "issue",
                 "permit", "entire", "exceed", "failure",
                 "100", "author", "rely"]

    # Contratto di esempio + Reserved Keywords
    contract_demo = """ pragma solidity ^0.5.0;

                    contract C {
                       //private state variable
                       uint private data;

                       //public state variable
                       uint public info;

                       //constructor
                       constructor() public {
                          info = 10;
                       }
                       //private function
                       function increment(uint a) private pure returns(uint) { return a + 1; }

                       //public function
                       function updateData(uint a) public { data = a; }
                       function getData() public view returns(uint) { return data; }
                       function compute(uint a, uint b) internal pure returns (uint) { return a + b; }
                    }
                    //External Contract
                    contract D {
                       function readData() public returns(uint) {
                          C c = new C();
                          c.updateData(7);         
                          return c.getData();
                       }
                    }
                    //Derived Contract
                    contract E is C {
                       uint private result;
                       C private c;

                       constructor() public {
                          c = new C();
                       }  
                       function getComputedResult() public {      
                          result = compute(3, 5); 
                       }
                       function getResult() public view returns(uint) { return result; }
                       function getData() public view returns(uint) { return c.info(); }

                abstract 	after 	alias 	apply
                auto 	case 	catch 	copyof
                default 	define 	final 	immutable
                implements 	in 	inline 	let
                macro 	match 	mutable 	null
                of 	override 	partial 	promise
                reference 	relocatable 	sealed 	sizeof
                static 	supports 	switch 	try
                typedef 	typeof 	unchecked mapping """
    demo_words = {word.lower() for word in re.split(r'\W+', contract_demo) if word}
    stopwords = nltk.corpus.stopwords.words("english")

    def __init__(self, file):
        """ :param file: percorso del file csv dal quale estrarre i contratti """

        self.file = file

        self.lemmatized_documents = [self.lemma_tokens(doc) for doc in self.contracts_extraxtor()]
        self.stemmed_documents = [self.stem_tokens(doc) for doc in self.contracts_extraxtor()]

        """ Creazione datasets e file csv """
        data = {"contratti": [i for i in self.lemmatized_documents]}
        self.lemma_dataset = pd.DataFrame(data=data)
        self.lemma_dataset.to_csv(path_or_buf="Deposito_contratti/cleaned_dataset_lemma.csv",
                                  index=False, header=False)

        data = {"contratti": [i for i in self.stemmed_documents]}
        self.stem_dataset = pd.DataFrame(data=data)
        self.stem_dataset.to_csv(path_or_buf="Deposito_contratti/cleaned_dataset_stem.csv",
                                 index=False, header=False)

    def contracts_extraxtor(self):
        """ Apre il file contratti in modalità lettura e restituisce una lista
            contenente tutti i contratti del file. Attua una prima pulizia eliminando
            gli underscore presenti all' inizio e alla fine di alcune parole

            :return: set contenente i contratti estratti"""
        # TODO: poi mettere una tupla
        contratti = set()  # utilizziamo un set per escludere eventuali doppioni di contratti
        with open(self.file, "r", encoding="utf-8") as reader:
            frase = ""
            for row in reader:
                for letter in row:
                    if letter != "_":
                        frase += letter
                        if letter == "þ":
                            """ þ è il separatore dei contratti """
                            contratti.add(frase)
                            frase = ""
                    else:
                        frase += " "
                        pass

        return contratti

    def cleaning(self, doc):
        """ Processo di pulizia dei documenti + gestione parole in CamelCase.

            :param doc: documento/contratto da pulire
            :return: lista di parole contenute nel documento ripulito da punteggiatura e CamelCase"""

        raw_words = [word.lower() for word in re.split(r"\W+", doc) if len(word) > 1]  # prima pulizia dei testi

        """ Gestione parole in CamelCase """
        camel = re.findall(r"([a-zA-Z]+[A-Z]+[a-z]+)+", doc)  # individua i CamelCases

        scamelled_case = ""
        for word in camel:
            for letter in word:
                if letter != letter.upper():  # individua l' inizio della nuova parola
                    scamelled_case += letter
                else:
                    scamelled_case += " " + letter
            scamelled_case += " "

        for word in scamelled_case.split():
            raw_words.append(word.lower())
        return raw_words

    def lemma_tokens(self, doc):
        """ Questa funzione ci permette di attuare un processo di Lemmatizzazione
            ai documenti che le passiamo.

            :param doc: documento/contratto da pulire
            :return: testi dei contratti puliti e lemmatizzati
            """

        raw_words = self.cleaning(doc)
        wnl = WordNetLemmatizer()
        lista_lemma = [wnl.lemmatize(word, 'v') for word in raw_words if len(wnl.lemmatize(word, 'v')) > 1]

        tokens = [word for word in lista_lemma if word not in self.bad_words]
        tokens = [word for word in tokens if word not in self.stopwords]
        tokens = [word for word in tokens if word not in self.demo_words]
        tokens = [word for word in tokens if wn.synsets(word)]

        # Ricostruiamo i testi dei contratti mettendo insieme le parole pulite
        cleaned_text_lemma = ""
        for word in tokens:
            cleaned_text_lemma += " " + word
        return cleaned_text_lemma  # ritorna i testi dei contratti puliti

    def stem_tokens(self, doc):
        """ Questa funzione ci permette di attuare un processo di Stemming
            ai documenti che le passiamo.

            :param doc: documento/contratto da pulire
            :return: testi dei contratti puliti e lemmatizzati
            """
        raw_words = self.cleaning(doc)
        tokens = [word for word in raw_words if word not in self.bad_words]
        tokens = [word for word in tokens if word not in self.stopwords]
        tokens = [word for word in tokens if word not in self.demo_words]
        tokens = [word for word in tokens if wn.synsets(word)]

        # lancaster = nltk.LancasterStemmer()
        porter = nltk.PorterStemmer()  # rimuove il suffisso

        lista_porter = []
        for t in tokens:
            if len(porter.stem(t)) > 1:
                lista_porter.append(porter.stem(t))

        """ Ricostruiamo i testi dei contratti mettendo insieme le parole pulite """
        cleaned_text_porter = ""
        for word in lista_porter:
            cleaned_text_porter += " " + word

        return cleaned_text_porter  # ritorna i testi dei contratti puliti


if __name__ == "__main__":
    test = DatasetPulito("Deposito_contratti/contracts.csv")
