# An end-to-end ranking system based on customers reviews: Integrating semantic mining and MCDM techniques
Welcome. This repository contains the python based implementation of the 'An end-to-end ranking system based on customers reviews: Integrating semantic mining and MCDM techniques'. In this repository, source codes of our paper are presented in terms of offline and online phases.  In corresponded paper, we propose an end-to-end ranking method for integrating mechanisms such as text processing, sentiment analysis and the multi-criteria decision-making technique. The proposed ranking method relies on the integration of three methods, namely, the aspect-based sentiment analysis (ABSA) method, the Dawid-Skene algorithm and the Best Worst Method (BWM). In other words, the proposed work encompasses four major steps: i) crawling customer reviews, ii) preprocessing, iii) aspect term extraction, aspect category detection and polarity detection, and iv) designing a decision-making model.  

# Notes
In this repository, there are three folders namely #Scrap, #online and #offline phases. In each folder implementation of correspond phase is presented. not that before run codes, ypu must download Glove 6b.200 pretrained word2wec model from https://nlp.stanford.edu/projects/glove/ and paste it in the data folder which exist in online and offline folders. Also, crawled data which are collected from https://www.tripadvisor.com are existed in crawled_data folder. This data are collected using implemented codes exist in #Scrape folder and due to possible changes in the site structure, they should be changed according to the latest site changes before implementation.
 

# Citation
If you find this code useful please cite us in your work:

Milad Eshkevari, Mustafa Jahangoshai Rezaee, Morteza Saberi, Omar K. Hussain,
An end-to-end ranking system based on customers reviews: Integrating semantic mining and MCDM techniques,
Expert Systems with Applications,
Volume 209,
2022,
118294,
ISSN 0957-4174,
https://doi.org/10.1016/j.eswa.2022.118294.
(https://www.sciencedirect.com/science/article/pii/S0957417422014294)

