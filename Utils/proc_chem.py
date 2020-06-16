# -*- coding: utf-8 -*-
"""
Created on  June 10th
@author: hanshanley

This file handles the processing of chemcial features of MOLs and SMILES. Specifically,
it holds the RDKit functions that determine the features of SMILES and functions that 
randomize SMILES.
"""

import  random
import 	gzip
import  re
import  functools

def to_mol(smi):
    """
    Creates a Mol object from a SMILES string.
    :param smi: SMILES string.
    :return: A Mol object or None if it's not valid.
    """
    if smi:
        return rkc.MolFromSmiles(smi)

def to_smiles(mol):
    """
    Converts a Mol object into a canonical SMILES string.
    :param mol: Mol object.
    :return: A SMILES string.
    """
    return rkc.MolToSmiles(mol, isomericSmiles=False)

def randomize_smiles_string(smile, random_type= 'restricted'):
	"""
	Returns a random SMILES given a SMILES of a molecule.
	:param smile: A string object
	:param random_type: The type (unrestricted, restricted) of randomization performed.
	:return : A random SMILES string of the same molecule or None if the molecule is invalid.
	"""
	if not smile:
		return None

	mol = to_smiles(smile)

	if random_type == "unrestricted":
		return rkc.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
	if random_type == "restricted":
		new_atom_order = list(range(mol.GetNumAtoms()))
		random.shuffle(new_atom_order)
		random_mol = rkc.RenumberAtoms(mol, newOrder=new_atom_order)
		return rkc.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
	raise ValueError("Type '{}' is not valid".format(random_type))