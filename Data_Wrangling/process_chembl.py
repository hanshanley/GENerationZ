# -*- coding: utf-8 -*-
"""
Created on  June 9th
@author: hanshanley

This program handles the preprocessing of the ChEMBL dataset. The ChEMBL dataset contains the 
chemical information about bioactive chemical compounds. The dataset subset that we consider
consists of all the 'small moleculues. This original dataset contains 1.9M different chemical
compounds. In this preprocessing step we remove the chemical compounds that do not contain 
valid SMILES strings. We further create a dataset with all the valid chmbl strings and
and their corresponding deepsmiles. Information on deepsmiles can be found at:
https://github.com/baoilleach/deepsmiles
"""

import csv 
import pandas as pd
import sys, os
import deepsmiles


## Read the  ChEMBL dataset
cur_path = os.path.dirname(__file__)
data_path = os.path.relpath('../Datasets/CHEMBL27-chembl_27_molecule-upFpv_RO77rZ-8A9RHrrh-86bsuI-i9aXM3g2pFroWM=.csv', cur_path)
f_data_path = os.path.relpath('../Datasets/fChEMBL_Smiles.csv', cur_path)
fds_data_path = os.path.relpath('../Datasets/fdsChEMBL_Smiles.csv', cur_path)

if (os.path.isfile(f_data_path)== False):
	chembl = pd.read_csv(data_path, sep =';')

	## Drop rows without Smiels strings 
	chembl.dropna(subset=['Smiles'], inplace=True)
	# Save chmebl subset to a csv file 

	## Get all the relevant smiles
	chembl_smiles = chembl['Smiles']
	chembl_smiles.to_csv(f_data_path,header=True,index=False)
chembl_smiles = pd.read_csv(f_data_path,header=0,sep =';')

## Get all the relevant deepsmiles and add them
## to the panda
deep_smiles = []
print("DeepSMILES version: %s" % deepsmiles.__version__)
converter = deepsmiles.Converter(rings = True, branches = True)
print(converter) # record the options used
for i in range(len(chembl_smiles)):
	deep_smiles.append(converter.encode(chembl_smiles['Smiles'][i]))
	if(i%100000== 0):
		print(i)
print(len(chembl_smiles))
print(len(deep_smiles))
chembl_smiles['Deep Smiles'] = deep_smiles
chembl_smiles.to_csv(fds_data_path,header=True,index=False)

## Ensure that Smiles and the Deep Smiles correpsond 
## uniquely to one another and that deep smiles can 
## be converted back to ordinary smiles 
for i in range(len(chembl_smiles)):
	try:
		decoded_smile = converter.decode(chembl_smiles['Deep Smiles'][i])
		if decoded_smile != chembl_smiles['Smiles'][i]:
			print(' Decoded Deep Smile was not equal to Original Smile')
	except deepsmiles.DecodeError as e:
		print("Decode Error! Error message was '%s'" % e.message)





