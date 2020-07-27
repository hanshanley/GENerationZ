# Private-De-Novo-Drug-Creation

We train a convolutional variational autoencoder (CVAE) to convert three different types of discrete representations (SMILES, DEEP SMILES, SELFIES) of drug-like molecules to and from a multidimensional latent space. We make use of advance scheduling technique proposed by Fu et al. to avoid posterior collapse in our variational autoencoder, allowing highly meaningful features to be extraxted from drug-like molecule. We found that using this method the latent space is found be grouped by QED, logP, and Synthensizability Access Scores (SAS).

We secondly train a CVAE with an auxillary network to shape the latent space to model a particular drug-like molecules efficacy (IC50) against a particular subset of targets. We model these targets using RNA seqences. We show that by shaping the latent space using this method we can mroe effecitvely predict IC50 scores from the latent representations and optimize molecules in the latent space to be more efficacious against a particular target.

We thirdly train seperate transformer decoder to map pretrained latent embeddings of molecules back to the discrete representations. We show that this separate more powerful decoder can effectively and easily more diverse sets of latent representations back to the latent space. As a result we enable our model to more efficiently decode optimized and 'dead' regions of the latent space. 

This work was done in completion of the MSc in Statistical Science at the University of Oxford. 
