# shor_multi_chip
Computation of resource needed to run shor on a surface code distributed architecture.

Code related to the preprint https://arxiv.org/abs/2504.08891

This code evaluate the logical error rate of a distributed rotated surface code patch. 
Then it uses this model to estimate the resources required for factoring 2048-bit RSA number in a chain-like architecture of processors using lattice surgery.

More precisely:

  - Multi_chip_threshold folder contain all the element to do the fit on the logical error rate per round of a distributed patch as well as the data from our simulations
    Note that in this folder the file to execute is the "Main_notebook.ipynb" file.
  - Resource_estimator contains all the code to estimate the resources required for Shor given the model for the logical error rate per round.
    Note that in this folder the file to execute is cout_shor.py
  - generate_factory is the code to generate the different factories we used in the resource estimation using the resource estimation code of https://arxiv.org/abs/1905.06903

/!\ to be used, generate_factory require to download the code of the article https://arxiv.org/abs/1905.06903 available on github.
