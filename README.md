# GENerateZ: Designing anticancer drugs using transcriptomic data, variational autoencoders, and genetic algorithmsPrivate-De-Novo-Drug-Creation

We propose a novel machine learning architecture and technique for de novo drug discovery of anti-cancer drugs by using discrete representations of drugs’ chemical compositions and the transcriptomics data of targets. In particular, we generate novel compounds optimized for high efficacy against specific types of cancerous cells. 

First, we train a variational autoencoder (VAE)  to convert discrete representations (SELFIES) of drug-like molecules to and from a multidimensional latent space. By utilizing the advanced scheduling technique cyclical annealing proposed by Fu et al., we avoid VAE posterior collapse. This advancement allows the extraction of highly meaningful latent features from our discrete representations. As proof of the latent space’s meaningfulness, we discover the distribution of molecules in the latent space follows a gradient of desirable chemical properties, namely the Quantitative Estimate of Drug-Likeness and Synthetic Accessibility; (molecules with high values are located in one region of the latent space, and molecules with low values are in another). These properties are important criterion for pre-clinical drug discovery and discovering this gradient allows for efficient optimization. 

Second, we jointly train another VAE with a separate auxiliary network designed to predict drug-like molecules' efficacy IC50 against a particular subset of cancer cell targets from their latent representations and the transcriptomic profiles of cancer cells. This joint training causes the latent space to develop gradients for a particular drug-like molecules' efficacy (IC50) against these particular targets. We discover that by utilizing this approach, we can accurately predict IC50 efficacy from VAE latent representations and transcriptomic data with an overall Pearson correlation of 86.78 on a test set. Using this approach, we further show following the gradient in the latent space using Bayesian optimization and genetic algorithms, we can optimize drug-like molecules’ efficacy against a particular cancer target. We illustrate the usefulness of this approach by generating hundreds of  novel potent drug candidates, optimized for efficacy against a group of sarcoma cells.

We verify these novel candidate drugs by comparing them to existing compounds with known efficacy against corresponding cancer type. We give an example an optimized drug:

CCC1=C2C=C(C=NC2=NC3=C1ON4C3=CC5=C(C4=O)COC(=O) C5(CC)O)C

<img src="https://github.com/hanshanley/Private-De-Novo-Drug-Creation/blob/master/SARCOMA_GA_IC50_SMILES_2 (1).png" width="480">


Last, we train a separate Transformer decoder to map pretrained latent embeddings of molecules back to their chemical representations. We show that this separate and more powerful decoder can more effectively and easily map diverse sets of latent representations back into compounds than the original decoder in the VAE. This essentially turns our VAE into a highly efficient feature extractor and allows for more exploration of the chemical space created by the VAE. 

We wish for our approach, which leverages the latest improvements in text generation from natural language processing, to be a step toward improving success rates of targeted and personalized drug discovery against cancers.


