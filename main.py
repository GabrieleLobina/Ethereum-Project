from Scraping import ObtainContracts

if __name__ == "__main__":
    extraction_w_Chrome = ObtainContracts(["https://etherscan.io/contractsVerified/1?ps=100",
                                           "https://etherscan.io/contractsVerified/2?ps=100",
                                           "https://etherscan.io/contractsVerified/3?ps=100",
                                           "https://etherscan.io/contractsVerified/4?ps=100",
                                           "https://etherscan.io/contractsVerified/5?ps=100"])
