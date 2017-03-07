A host-pathogen protein interaction prediciton tool using a combination of the stochastic gradient descent method and the soft confident-weighted method [1].

How to run:
Download training data from "http://sunflower.kuicr.kyoto-u.ac.jp/jira/ppi_prediction/data.zip"
Extract the training data and place the folder data in the same directory as predict.py and multilayernn.py
Run
    python predict.py species c eta 
where species is the species of pathogen in the prediction, 'Styphi' 'Ftularensis' 'Banthracis' and 'Ypestis',
c and eta are parameters of the soft confidence-weighted method.


[1] Wang, Jialei, Peilin Zhao, and Steven CH Hoi. "Soft Confidence-Weighted Learning." ACM Transactions on Intelligent Systems and Technology (TIST) 8.1 (2016): 15.
