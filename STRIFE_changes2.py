#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 17:12:05 2021

@author: Tom Hadfield

Main class for STRIFE algorithm

python STRIFE_changes2.py -p Mpro.pdb -f less_fragments_to_elaborate2.sdf -o otw -s less_fragments_to_elaborate2.smi -it otw -m new_hotspots100


"""



#########Standard Libraries##########
from curses.panel import top_panel
from email import header
import json
import time
import argparse
import os
from tkinter import E
from IPython import embed
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning, module = 'tensorflow')

from elaborations import elaborate
import numpy as np
import pandas as pd
import multiprocessing as mp #For parallelising
import time
import matplotlib.pyplot as plt
import sys 
from random import sample
from functools import partial
import glob
import openbabel
from datetime import datetime
from data_prep.specifyExitVector import addDummyAtomToMol
import sascorer
#########RDKit Modules############
import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import QED
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMMPA
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole 
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') #Supress annoying RDKit output

import pickle



from docking import docking
#Create feature factory
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
from preprocessing import preprocessing
from hotspots.hs_io import HotspotReader




os.system("taskset -p 0xff %d" % os.getpid())

class STRIFE:
    
    
    def __init__(self, args):
        #run 'python STRIFE.py -h' for definitions of the arguments
        
        
        assert bool(args.protein) == 1, 'Please specify the path to a PDB file'
        assert bool(args.fragment_sdf) == 1, 'Please specify the location of the fragment SDF. This can also be an SDF of a larger ligand of which the fragment is a substructure'
        
        #Convert the provided paths to the absolute path 
        args.protein = os.path.abspath(os.path.expanduser(args.protein))
        args.fragment_sdf = os.path.abspath(os.path.expanduser(args.fragment_sdf))

        #Check that the arguments exist
        
        if args.protein is None:
            raise ValueError('You must specify a pdb file as the protein')
        else:
            assert os.path.exists(args.protein), f'Specified protein file, {args.protein}, does not exist'
        
        if args.fragment_sdf is None:
            raise ValueError('You must specify an SDF file, either containing the molecule to be used as a fragment, or a superstructure of it')
        else:
            assert os.path.exists(args.fragment_sdf), f'Specified fragment SDF file, {args.fragment_sdf}, does not exist'
        
        #If the output directory doesn't exist, create it
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)

        args.output_directory = os.path.abspath(os.path.expanduser(args.output_directory))




        if args.fragment_smiles is not None:
            
            #Check whether args.fragment_smiles is a file
            if os.path.exists(os.path.expanduser(args.fragment_smiles)):
                args.fragment_smiles = os.path.abspath(os.path.expanduser(args.fragment_smiles))
                smiles_file = True
            else:
                #Check that we can parse it with RDKit
                try:
                    if Chem.MolFromSmiles(args.fragment_smiles).GetNumHeavyAtoms() > 0:
                        smiles_file = False
                    else: 
                        raise ValueError('Fragment must have at least one heavy atom')
                except:
                    
                    raise ValueError("The supplied fragment_smiles doesn't appear to be a file and RDKit is unable to parse it as a molecule. Please check that you're providing a valid fragment")
        
            if smiles_file:
                with open(args.fragment_smiles, 'r') as f:
                    fragSmiles = f.read()
            
            else:
                fragSmiles = args.fragment_smiles
            
        
        else:
            #using exit_vector_idx instead
            mol = Chem.SDMolSupplier(args.fragment_sdf)[0]
            molWithDummyAtom = addDummyAtomToMol(mol, args.exit_vector_idx)
            fragSmiles = Chem.MolToSmiles(molWithDummyAtom)
        
        
        
        if args.write_elaborations_dataset:
            self.writeFinalElabs = True
        else:
            self.writeFinalElabs = False
        
        
     

        
        #Check that we're inputting a valid pharmacophoric representation

        
       
        self.storeLoc = args.output_directory
        

        if args.num_cpu_cores > 0:
            self.num_cpu_cores = args.num_cpu_cores
        elif args.num_cpu_cores == -1:
            self.num_cpu_cores = mp.cpu_count()
        else:
            raise ValueError("Please supply a valid number of cores to use, or specify num_cpu_cores as -1 to use all available cores")


        #Create subclasses
        self.elaborate = elaborate()
        self.docking = docking()
        self.preprocessing = preprocessing()
        
        


        
        if bool(args.load_specified_pharms):
            self.hotspotStore = args.load_specified_pharms

            #Load the specified pharmacophoric points from args.output_directory
            if len(glob.glob(f'{self.hotspotStore}/acceptorHotspot.sdf')) + len(glob.glob(f'{self.hotspotStore}/donorHotspot.sdf')) == 0:
                print('If manually specifying pharmacophoric points, please provide at least one point\n')
                print(f'Donors should be saved in the file {self.hotspotStore}/donorHotspot.sdf\n')
                print(f'Acceptors should be saved in the file {self.hotspotStore}/acceptorHotspot.sdf\n')
                raise ValueError('If manually specifying pharmacophoric points, please provide at least one point')
                
            elif len(glob.glob(f'{self.hotspotStore}/acceptorHotspot.sdf')) == 0:
                hotspotsDict = {}
                hotspotsDict['Acceptor'] = Chem.RWMol()
                hotspotsDict['Donor'] = Chem.SDMolSupplier(f'{self.hotspotStore}/donorHotspot.sdf')[0]
            elif len(glob.glob(f'{self.hotspotStore}/donorHotspot.sdf')) == 0:
                hotspotsDict = {}
                hotspotsDict['Acceptor'] = Chem.SDMolSupplier(f'{self.hotspotStore}/acceptorHotspot.sdf')[0]
                hotspotsDict['Donor'] = Chem.RWMol()
            else:
                hotspotsDict = {}
                hotspotsDict['Acceptor'] = Chem.SDMolSupplier(f'{self.hotspotStore}/acceptorHotspot.sdf')[0]
                hotspotsDict['Donor'] = Chem.SDMolSupplier(f'{self.hotspotStore}/donorHotspot.sdf')[0]
        
        
        

        
        
        #Set up the hotspotsDF
        self.HPositions = [] #list for hotspot positions
        self.HType = [] #list for hotspot type (donor/acceptor)
        #self.Distances = [] #Distance from exit vector
        #self.Angles = [] #Angle from exit vector
        self.origAtomIdx = []
        self.scoresList = []
        self.Scores = []
        for pharm in ['Acceptor', 'Donor']:

            if hotspotsDict[pharm].GetNumHeavyAtoms() > 0: #Changed self.hotspotsDict to hotspotsDict here
                self.scoresList.append(np.array((hotspotsDict[pharm].GetProp('vdw').split())))
                for atom in hotspotsDict[pharm].GetAtoms():
                    pos = np.array(hotspotsDict[pharm].GetConformer().GetAtomPosition(atom.GetIdx()))

                    self.HPositions.append(pos)
                    #self.Distances.append(self.preprocessing.vectorDistance(pos, self.exitVectorPos))
                    #self.Angles.append(self.preprocessing.vectorAngle(pos, self.exitVectorPos))
                    self.HType.append(pharm)
                    self.origAtomIdx.append(atom.GetIdx()) #Atom index so we can recover it if necessary

        for sublist in self.scoresList:
            for score in sublist:
                self.Scores.append(float(score))
        
        self.HotspotsDF = pd.DataFrame({'position':self.HPositions, 'type':self.HType, 'score': self.Scores }).sort_values('score', ascending = False).reset_index(drop = True)
        
        if args.iter_type == 'distance' or args.iter_type == 'score': #or args.iter_type == 'otw':
            #rule out the least important hotspots that are within 4A of each other
            self.Filtered=True
            if self.Filtered:
                self.FilteredDF = pd.DataFrame()
                self.FilteredDF = self.FilteredDF.append(self.HotspotsDF.loc[self.HotspotsDF['score']== max(self.HotspotsDF['score'])])
                for idx, row in self.HotspotsDF.iterrows():
                    add = 1
                    for idx2, row2 in self.FilteredDF.iterrows():
                        if self.preprocessing.vectorDistance(row['position'], row2['position']) < 3:
                            add = 0
                    if add == 1:
                        self.FilteredDF = self.FilteredDF.append(row)

                self.HotspotsDF = self.FilteredDF
                self.HotspotsDF = self.HotspotsDF.reset_index(drop=True)
            #embed(header = '242')
            
        #tell it to rule out hotspots below a certain score..
        self.HotspotsDF = self.HotspotsDF[self.HotspotsDF['score'] > 0.1]
        self.HotspotsDF = self.HotspotsDF.reset_index(drop=True)





        #Import pathLength classifier
        with open(args.path_length_model, 'rb') as f:
            self.clf = pickle.load(f)
            
    
    def run(self, args):
        self.satisfiedHotspots = pd.DataFrame({})
        self.fragMol3D = Chem.SDMolSupplier(args.fragment_sdf)[0]
        temp_hotspots_df = self.HotspotsDF
        
        #check if any hotspots are already satisfied by the fragment
        for indx2 in range(len(self.HotspotsDF)):
            distanceToPharm = []
            feats = factory.GetFeaturesForMol(self.fragMol3D)
            for feat in feats:
                if feat.GetFamily() == self.HotspotsDF['type'][indx2]:
                    
                    #Compute Distance to the  hotspot and take the pharmacophore with the smallest distance
                    pharmPosition = np.array(self.fragMol3D.GetConformer().GetAtomPosition(feat.GetAtomIds()[0]))
                    distanceToPharm.append(self.preprocessing.vectorDistance(pharmPosition, self.HotspotsDF['position'][indx2]))
            if len(distanceToPharm) != 0:


                if np.min(distanceToPharm) < 3.5:
                    #then constraint is already satisfied
                    #add constraint to list of hotspots that should be satisfied

                    self.satisfiedHotspots = self.satisfiedHotspots.append(self.HotspotsDF.iloc[[indx2]]).reset_index(drop=True)
                    
                    temp_hotspots_df = temp_hotspots_df.drop(indx2)
        #self.HotspotsDF = temp_hotspots_df.reset_index(drop=True)
        self.HotspotsDF = temp_hotspots_df

        



        #the actual loop of making elaborations to as many hotspots as possible
        for index in range(len(self.HotspotsDF)):
            
            if len(self.HotspotsDF) == 0:
                
                break


            #if this is the first elaboration, take the fragment supplied
            if index==0:
                self.fragMol3D_whole = self.fragMol3D
                self.HotspotsDF = self.HotspotsDF
            #but if not take the previous elaboration
            else:
                self.fragMol3D = self.fragMol3D_updated
                self.HotspotsDF = self.HotspotsDF_updated

            #check SA score of adding to each atom in the fragment
            self.avoid=[]
            self.orig_score = sascorer.calculateScore(self.fragMol3D)
            #first calculate the SA of the fragment
            
            #dummyAtomIdx = rwmol.AddAtom(Chem.Atom(0)) #Add dummy atom and get its idx
            #rwmol.AddBond(atomIdx, dummyAtomIdx, Chem.BondType.SINGLE)
            
            #Chem.SanitizeMol(rwmol)
            #then compare this to the SA of the fragment with a benzene ring added on to it at each atom
            
            benz = Chem.MolFromSmiles('Cc1ccccc1')
            mol_rw = Chem.RWMol(Chem.CombineMols(benz, self.fragMol3D))

            for atom in self.fragMol3D.GetAtoms():
                index2 = atom.GetIdx()
                mol_rw = Chem.RWMol(Chem.CombineMols(benz, self.fragMol3D))
                mol_rw.AddBond(0, 7+index2, order = Chem.rdchem.BondType.SINGLE )
            
                #try:
                done_mol = mol_rw.GetMol()
                try:
                    Chem.SanitizeMol(done_mol)
                    
                    if sascorer.calculateScore(done_mol) > 1.5*self.orig_score:
                        self.avoid.append(index2)
                except:
                    self.avoid.append(index2)
            #return the index of the exit vector, also define self.topHotspot
            if args.iter_type == 'score' or args.iter_type == 'otw':
                self.exitVectorIndex = self.chooseExitVector(self.fragMol3D, self.HotspotsDF, index, self.avoid)
            elif args.iter_type == 'distance':
                self.exitVectorIndex = self.chooseExitVectorDistance(self.fragMol3D, self.HotspotsDF, index, self.avoid)   

            try:
  
                molWithDummyAtom = addDummyAtomToMol(self.fragMol3D, self.exitVectorIndex)
            except:
                #embed(header='except line 319')
                #didn't like adding that atom there
                self.avoid.append(self.exitVectorIndex)
                self.exitVectorIndex = self.chooseExitVector(self.fragMol3D, self.HotspotsDF, index, self.avoid)
                try:    
                    molWithDummyAtom = addDummyAtomToMol(self.fragMol3D, self.exitVectorIndex)
                except:
                    #just skip this hotspot
                    #self.HotspotsDF = self.HotspotsDF.drop(self.topHotspot.index).reset_index(drop=True)
                    
                    if len(self.HotspotsDF) > 0:
                        self.HotspotsDF = self.HotspotsDF.drop(self.topHotspot.index)
                        self.fragMol3D_updated = self.fragMol3D
                        self.HotspotsDF_updated = self.HotspotsDF
                        continue
                    else:
                        break

            fragSmiles = Chem.MolToSmiles(molWithDummyAtom)

            #not touching this
            print('Preprocessing fragment')
            #Store fragment SMILES in the output directory:
            with open(f'{self.storeLoc}/frag_smiles{index}.smi', 'w') as f:
                f.write(fragSmiles)

            with open(f'{self.storeLoc}/frag_smiles.smi', 'w') as f:
                f.write(fragSmiles)
            fc, evI, evp, fc2 = self.preprocessing.preprocessFragment(fragSmiles, self.fragMol3D)

            if args.iter_type == 'otw':
                #got exit vector position, evp
                #find exit vector - top hotspot
                hotspots_to_satisfy = []
                self.tempTopHotspot = pd.DataFrame()
               
                exit_tophotspot = list(self.topHotspot['position'])[0] - evp
                conf = self.fragMol3D.GetConformer()
                for idx in self.HotspotsDF.index.tolist():
                    secondary_hotspot = self.HotspotsDF['position'][idx]
           
                    if self.HotspotsDF['score'][idx] == float(self.topHotspot['score']):
                        #can't have top hotspot as the secondary on the way hotspot
                        #that would be silly!
                        continue
                    else:
                       
                        exit_secondary = secondary_hotspot - evp
                        print(exit_secondary)
                        angle = (self.preprocessing.vectorAngle(exit_secondary,exit_tophotspot))
                        if angle < np.pi/5 : #possibly should be pi/6
                            
                            if self.preprocessing.plausibilityScore(secondary_hotspot, self.possExitVector['position'],self.possExitVectorNeighbor) != 1:
                                continue
                            else:
                                self.tempTopHotspot = self.tempTopHotspot.append(self.HotspotsDF.loc[idx,:])
                            #then elaborate to this and the top hotspot simultaneously
                if len(self.tempTopHotspot) != 0:
                    for indx in self.tempTopHotspot.index.to_list():
                        self.topHotspot = self.topHotspot.append(self.tempTopHotspot.loc[indx,:])
                print('and we are going to elaborate to...')
                print(self.topHotspot)
  


            Chem.MolToMolFile(fc, f'{self.storeLoc}/frag_{index}.sdf')
            #Save fragment SDF
            Chem.MolToMolFile(fc, f'{self.storeLoc}/frag.sdf')
            #Save constraint SDF (will need to be converted to Mol2 using obabel)
            Chem.MolToMolFile(fc2, f'{self.storeLoc}/constraint.sdf')

            #Save fragment exit position
            np.savetxt(f'{self.storeLoc}/evp.txt', evp)
            #Convert the constraint.sdf file to constraint.mol2 (for constrained docking in GOLD)
            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("sdf", "mol2")
            mol = openbabel.OBMol()
            obConversion.ReadFile(mol, f'{self.storeLoc}/constraint.sdf')
            obConversion.WriteFile(mol, f'{self.storeLoc}/constraint{index}.mol2')
            




            
            self.exitVectorPos = evp
            self.frag = f'{self.storeLoc}/frag_smiles.smi'
            self.fragCore = fc
            self.constraintFile = f'{self.storeLoc}/constraint{index}.mol2'
            self.cavityLigandFile = args.fragment_sdf #Used for docking in GOLD to define the binding pocket
            self.protein = args.protein


            #Set up the hotspotsDF
            
            if len(self.topHotspot) > 1:

                #for index in range(len(self.topHotspot)/3):
                dists = []
                angs = []
                for indx in self.topHotspot.index.to_list():
                   
                    dists.append(self.preprocessing.vectorDistance(self.topHotspot['position'][indx],self.exitVectorPos) -3)
                    angs.append(self.preprocessing.vectorAngle(self.topHotspot['position'][indx],self.exitVectorPos))

                self.finalTopHotspot = pd.DataFrame({'distFromExit':dists, 'angFromExit':angs, 'position':list(self.topHotspot['position']), 'type':list(self.topHotspot['type']), index: self.topHotspot.index.to_list()})
                self.hMulti = self.preprocessing.prepareProfiles(self.finalTopHotspot, single = False)

                #do the actual thing
                self.finishedElabs, self.HotspotsDF_updated = self.elaborationsWithoutRefinement(counts = True, totalNumElabs = args.number_elaborations, numElabsPerPoint = args.number_elaborations_exploration, index=index, single=False)

                #check it returned something
                if [self.finishedElabs, self.HotspotsDF_updated] == [0,0]:
                    #embed(header = '418, didnt find any quasi actives')
                    #then the elaboration process didn't work, skip the hotspot
                    #self.HotspotsDF = self.HotspotsDF.drop(self.topHotspot.index).reset_index(drop=True)
                    self.HotspotsDF = self.HotspotsDF.drop(self.topHotspot.index)

                    self.fragMol3D_updated = self.fragMol3D
                    self.HotspotsDF_updated = self.HotspotsDF
                    #embed(header='414')
                    continue     
                else:           
                    self.fragMol3D_updated = self.finishedElabs[0]
                    self.QuasiActives = self.finishedElabs




            else:
                
                self.finalTopHotspot = pd.DataFrame({'distFromExit':self.preprocessing.vectorDistance(list(self.topHotspot['position'])[0],self.exitVectorPos), 'angFromExit':self.preprocessing.vectorAngle(list(self.topHotspot['position']),self.exitVectorPos), 'position':list(self.topHotspot['position']), 'type': list(self.topHotspot['type'])[0]}, index = [0])

                self.hSingles = self.preprocessing.prepareProfiles(self.finalTopHotspot)['0']


                self.finishedElabs, self.HotspotsDF_updated = self.elaborationsWithoutRefinement(counts = True, totalNumElabs = args.number_elaborations, numElabsPerPoint = args.number_elaborations_exploration, index=index)
                if [self.finishedElabs, self.HotspotsDF_updated] == [0,0]:
                    #then the elaboration process didn't work, skip the hotspot
                    #self.HotspotsDF = self.HotspotsDF.drop(self.topHotspot.index).reset_index(drop=True)
                    #embed()
                    if len(self.HotspotsDF) > 0:
                        self.HotspotsDF = self.HotspotsDF.drop(self.topHotspot.index)
                        self.fragMol3D_updated = self.fragMol3D
                        self.HotspotsDF_updated = self.HotspotsDF
                        #embed(header = '434')
                        continue     
                    else:
                        break
                else:           
                    self.fragMol3D_updated = self.finishedElabs[0]
                    self.QuasiActives = self.finishedElabs

            




    def chooseExitVector(self, fragment, hotspots, index, avoid = None):
        self.found_exit_vector = False

       
        while self.found_exit_vector == False and len(self.HotspotsDF) > 0:
            #top hotspot should be at the top of the list
           
            self.topHotspot = self.HotspotsDF.head(1)
            
            topHotspotCoords = list(self.topHotspot['position'])[0]
           
            #iterate through the atoms comprising the fragment to find the best exit vector
            fragConf = fragment.GetConformer()
            fragAtomIndices = []
            fragAtomPositions = []
            fragAtomDistances = []
            for atomIndex in range(fragment.GetNumHeavyAtoms()):
                fragAtomPos = (np.array(fragConf.GetAtomPosition(atomIndex)))
                fragAtomPositions.append(fragAtomPos)
                
                fragAtomDistances.append(self.preprocessing.vectorDistance(fragAtomPos, topHotspotCoords))
                fragAtomIndices.append(atomIndex)
            
            self.DummyDF = pd.DataFrame({'position': fragAtomPositions, 'distance': fragAtomDistances, 'atom_index':fragAtomIndices}).sort_values('distance').reset_index(drop=True)
            
            if len(avoid) != 0:
                for n in avoid:
                    self.DummyDF = self.DummyDF.drop(self.DummyDF[self.DummyDF['atom_index'] == n].index).reset_index(drop=True)
            #check that nearest atom is an appropriate selection to elaborate from
            self.possExitVector = pd.DataFrame(self.DummyDF.iloc[0])[0]
            self.possExitVectorIndex = int(self.possExitVector['atom_index'])
            atoms = fragment.GetAtoms()
            atom = atoms[self.possExitVectorIndex]

            #check that the atom indexing is working probably

            self.possExitVectorDegree = atom.GetDegree()

            if self.possExitVectorDegree == 1:
                #at the end of a chain, only need to check angle
                #find out nearest neighbour's index
                atom.GetNeighbors()[0].GetIdx()
                self.possExitVectorNeighbor = np.array(fragConf.GetAtomPosition(atom.GetNeighbors()[0].GetIdx()))
                self.possExitVectorPlausibility = self.preprocessing.plausibilityScore(topHotspotCoords, self.possExitVector['position'],self.possExitVectorNeighbor)
  
            elif self.possExitVectorDegree == 2:
                #could be aromatic..
                n1, n2 = atom.GetNeighbors()
                self.possExitVectorNeighbor = (np.array(fragConf.GetAtomPosition(n1.GetIdx())) + np.array(fragConf.GetAtomPosition(n2.GetIdx())))/2
                self.possExitVectorPlausibility = self.preprocessing.plausibilityScore(topHotspotCoords, self.possExitVector['position'],self.possExitVectorNeighbor)

            elif self.possExitVectorDegree == 3:
                #could be aromatic..
                n1, n2, n3 = atom.GetNeighbors()
                self.possExitVectorNeighbor = (np.array(fragConf.GetAtomPosition(n1.GetIdx())) + np.array(fragConf.GetAtomPosition(n2.GetIdx())) +np.array(fragConf.GetAtomPosition(n3.GetIdx())))/3
                self.possExitVectorPlausibility = self.preprocessing.plausibilityScore(topHotspotCoords, self.possExitVector['position'],self.possExitVectorNeighbor)


            elif self.possExitVectorDegree >= 3:
                self.possExitVectorPlausibility = 0
            
            #CHANGE BACK TO 1!!!!!!!!!!!!!!!!!!!!!!!!!
            if self.possExitVectorPlausibility == 1:
                #run using index to specify exit vector
                self.found_exit_vector = True
                

            else:
                #ignore that HOTSPOT (could change this to try second nearest exit vector!)
                #self.HotspotsDF = self.HotspotsDF.drop(self.topHotspot.index).reset_index(drop=True)
                self.HotspotsDF = self.HotspotsDF.drop(self.topHotspot.index)
        return self.possExitVectorIndex

            




    def chooseExitVectorDistance(self, fragment, hotspots, index, avoid = None):
        
        self.found_exit_vector = False
        while self.found_exit_vector == False and len(self.HotspotsDF) > 0:
            #need to calculate distances between all of the hotspots and all of the atoms in frag
            fragConf = fragment.GetConformer()
            self.distancesDF = pd.DataFrame(index =[atom.GetIdx() for atom in list(fragment.GetAtoms())], columns=self.HotspotsDF.index.tolist())
            

            for indx in self.HotspotsDF.index.tolist():
                hotspot_position = self.HotspotsDF['position'][indx]
                for atomIndex in range(fragment.GetNumHeavyAtoms()):
                    fragAtomPos = (np.array(fragConf.GetAtomPosition(atomIndex)))
                    
                    distance_hotspot_atom = self.preprocessing.vectorDistance(fragAtomPos,hotspot_position)
                    self.distancesDF[indx][atomIndex]= distance_hotspot_atom
            avoid = [1]
            if len(avoid) != 0:
                for n in avoid:
                    
                    self.distancesDF = self.distancesDF.drop(n).reset_index(drop=True)
            
            #now find minimum distance in dataframe
            #trickier than it should be?
            dict1 = {}
            for index in self.distancesDF.columns.tolist():
                dict1[self.distancesDF.min(axis=0)[index]] = index

            dict2 = {}
            for column in self.distancesDF.index.tolist():
                dict2[self.distancesDF.min(axis=1)[column]] = column

            min_distance = np.min(list(dict1.keys()))
            indx_best_hotspot = dict1[min_distance]
            indx_best_fragatom = dict2[min_distance]
           
            self.topHotspot = pd.DataFrame(self.HotspotsDF.loc[indx_best_hotspot]).T


            topHotspotCoords = list(self.topHotspot['position'])[0]

            #check that nearest atom is an appropriate selection to elaborate from
 
            self.possExitVectorIndex = indx_best_fragatom
            self.possExitVectorPosition = np.array(fragConf.GetAtomPosition(self.possExitVectorIndex))
            atoms = fragment.GetAtoms()
            atom = atoms[self.possExitVectorIndex]

            #check that the atom indexing is working probably

            self.possExitVectorDegree = atom.GetDegree()

            if self.possExitVectorDegree == 1:
                #at the end of a chain, only need to check angle
                #find out nearest neighbour's index
                atom.GetNeighbors()[0].GetIdx()
                self.possExitVectorNeighbor = np.array(fragConf.GetAtomPosition(atom.GetNeighbors()[0].GetIdx()))
                
                self.possExitVectorPlausibility = self.preprocessing.plausibilityScore(topHotspotCoords, self.possExitVectorPosition,self.possExitVectorNeighbor)
  
            elif self.possExitVectorDegree == 2:
                #could be aromatic..
                n1, n2 = atom.GetNeighbors()
                self.possExitVectorNeighbor = (np.array(fragConf.GetAtomPosition(n1.GetIdx())) + np.array(fragConf.GetAtomPosition(n2.GetIdx())))/2
                self.possExitVectorPlausibility = self.preprocessing.plausibilityScore(topHotspotCoords, self.possExitVectorPosition,self.possExitVectorNeighbor)


            elif self.possExitVectorDegree >= 3:
                self.possExitVectorPlausibility = 0
            
            #CHANGE BACK TO 1!!!!!!!!!!!!!!!!!!!!!!!!!
            if self.possExitVectorPlausibility == 1:
                #run using index to specify exit vector
                self.found_exit_vector = True
                

            else:
                #ignore that HOTSPOT (could change this to try second nearest exit vector!)
                #self.HotspotsDF = self.HotspotsDF.drop(self.topHotspot.index).reset_index(drop=True)
                self.HotspotsDF = self.HotspotsDF.drop(self.topHotspot.index)

        return self.possExitVectorIndex        
    def identifyQuasiActives(self):
        self.multiQuasiActives = {}
        self.multiQuasiActives = self.multiDistances.loc[self.multiDistances['distance'] < 3.5].drop_duplicates('smiles').head(5)

    def elaborationsWithoutRefinement(self, counts = True, totalNumElabs = 250, numElabsPerPoint = 250, n_cores = None,index=0, single=True):
        
        if n_cores is None:
            n_cores = self.num_cpu_cores


        
        self.singleElabs_noRefine = {}
        self.singleDocks_noRefine = {}
        self.singleDistances = {}
        self.elabsTestNoRefineDocksFiltered = {}


        if single == True:

            #iterate over the pharmacophoric points

            #set up HotspotSingle class
            self.singleElabs_noRefine = HotspotSingle(self.hSingles, self.frag, self.clf, self.constraintFile, self.cavityLigandFile, self.protein)

            if counts:
                #Make elaborations using the counts model
                #Here we don't filter on quality until the final set of elaborations has been sampled
                self.singleElabs_noRefine = self.elaborate.makeElaborationsNoFilter(self.singleElabs_noRefine, numElabsPerPoint=numElabsPerPoint, n_cores = self.num_cpu_cores)
            else:
                #Make elaborations using the Orig model 
                self.singleElabs_noRefine = self.elaborate.makeElaborationsNoFilter(self.singleElabs_noRefine, modelType = 'Orig', numElabsPerPoint=numElabsPerPoint, n_cores = self.num_cpu_cores)
    
                    #Now we want to take all of the elaborations we've made, sample some of them and then filter
            self.elabsTestNoRefine = pd.DataFrame()
            
            self.elabsTestNoRefine = self.elabsTestNoRefine.append(self.singleElabs_noRefine.profileElabs)

            #Now we have all of the molecules sampled from the quasi active profiles
            self.elabsTestNoRefineSample = self.elabsTestNoRefine.sample(n = totalNumElabs, random_state = 10) #Sample the number of elaborations we want
            self.elabsTestNoRefineFilter = self.elaborate.filterGeneratedMols(self.elabsTestNoRefineSample, n_cores = self.num_cpu_cores)
        
        
            #Now we want to dock these 
            self.elabsTestNoRefineFName = f'{self.storeLoc}/elabsTestNoRefine.sdf'
            self.elabsTestNoRefineCountsIdx, self.elabsTestNoRefineCountsSDFIdx = self.docking.getSDFs(self.elabsTestNoRefineFilter, self.fragCore, self.elabsTestNoRefineFName) 
            
            
            
            #Do docking
            self.elabsTestNoRefineDocks, self.elabsTestNoRefineFS = self.docking.dockLigandsMP(self.elabsTestNoRefineFName, self.constraintFile, self.cavityLigandFile, self.protein, returnFitnessScore = True, n_processes = self.num_cpu_cores)
            self.hSingles['position'] = list(self.hSingles['position'])
            
            self.singleDistances = self.docking.assessAllDocksNoRefinement(self.elabsTestNoRefineDocks, self.hSingles, True)
           
            self.elabsTestNoRefineDocksFiltered = self.singleDistances.loc[self.singleDistances['distance'] < 3.5].drop_duplicates('smiles')
            
            if len(self.elabsTestNoRefineDocksFiltered) == 0:
                #can't elabembeorate to this pharm


                return [0,0]
        
        
        
        
        elif single == False:
            
            self.multiElabs = HotspotMulti(self.hMulti, self.frag, self.clf, self.constraintFile, self.cavityLigandFile, self.protein)
            
            #Make elaborations using the counts model and filter to retain those with the desired pharmacophoric profile
            self.multiElabs = self.elaborate.makeElaborationsAndFilter(self.multiElabs, numElabsPerPoint=numElabsPerPoint, n_cores = n_cores)          
            #Prepare the filtered elaborations to be docked in GOLD
            self.multiElabs = self.docking.prepareForDocking(self.multiElabs, self.fragCore, f'{self.storeLoc}/countsElabsMulti.sdf')
            
            #do docking
            
            self.multiDocks, self.elabsTestNoRefineFS = self.docking.dockLigandsMP(self.multiElabs.dockingFname, self.constraintFile, self.cavityLigandFile, self.protein, returnFitnessScore = True, n_processes = n_cores) #Dock in parallel
            
            #Compute distance to pharmacophoric point
            self.multiDistances = self.docking.assessAllDocksNoRefinement(self.multiDocks, self.hMulti, single = False)
            
            self.elabsTestNoRefineDocksFiltered = self.multiDistances.loc[self.multiDistances['distance'] < 3.5].drop_duplicates('smiles')
            if len(self.elabsTestNoRefineDocksFiltered) == 0:
                #can't elaborate to this pharm

                return [0,0]

 ###########################
        #else:
        #if it satisfies the new pharm, add this to the satisfied list and check 
        #self.satisfiedHotspots = self.satisfiedHotspots.append(self.topHotspot).reset_index(drop=True)
        self.satisfiedHotspots = self.satisfiedHotspots.append(self.topHotspot)
        #put satisfied df into preprocessing friendly format
        distance_from_exit = []
        angle_from_exit = []
        
        for indx in self.satisfiedHotspots.index.to_list():
            distance_from_exit.append(self.preprocessing.vectorDistance(self.satisfiedHotspots['position'][indx], self.exitVectorPos))
            angle_from_exit.append(self.preprocessing.vectorAngle(self.satisfiedHotspots['position'][indx], self.exitVectorPos))
        self.satisfiedHotspotsFormatted = pd.DataFrame({'type' : list(self.satisfiedHotspots['type']), 'position':list(self.satisfiedHotspots['position']), 'distFromExit':distance_from_exit, 'angFromExit' : angle_from_exit})
        self.hMulti = self.preprocessing.prepareProfiles(self.satisfiedHotspotsFormatted)
    
        self.multiDistances = self.docking.assessAllDocksNoRefinement(self.elabsTestNoRefineDocksFiltered['mols'], self.hMulti, single=False)
        
        self.elabsTestNoRefineDocksFiltered =self.multiDistances.loc[self.multiDistances['distance'] < 3.5].drop_duplicates('smiles').reset_index(drop=True)
        #check again that this isn't empty
        if len(self.elabsTestNoRefineDocksFiltered) == 0:
            #can't elaborate to this pharm
            return [0,0]

        else:

            #output sdf of hotspots that claim to be satisfied
            
            df = self.satisfiedHotspots
            df = df.reset_index(drop=True)
            fakelig = 'I'
            for i in range(df.shape[0] - 1):
                fakelig += 'I'
            molPharmProf = Chem.MolFromSmiles(fakelig)
            
            #Now add the 3d position to each atom
            conf = Chem.Conformer(molPharmProf.GetNumAtoms())
            
            for idx, row in df.iterrows():
                conf.SetAtomPosition(idx, row['position'])
        
            conf.SetId(0)
            
            molPharmProf.AddConformer(conf)     

            Chem.MolToMolFile(molPharmProf,f'{self.storeLoc}/satisfiedHotspots_%s.sdf'%index)           
            #Compute ligand Efficiency
            #remake list of mols and docking scores

            self.mols_filt = []
            self.mols_filt_fs = []
            self.mols_scores = []
            for ind in range(len(self.elabsTestNoRefineDocksFiltered)):
                self.mols_scores.append(float(self.elabsTestNoRefineDocksFiltered['mols'][ind].GetProp('Gold.PLP.Fitness')))
                self.mols_filt.append(self.elabsTestNoRefineDocksFiltered['mols'][ind])
                self.mols_filt_fs.append(self.elabsTestNoRefineDocksFiltered['distance'][ind])
                self.elabsTestNoRefineDocksFiltered['mols'][ind].SetProp('Max_Pharm_Hotspot_Distance', str(self.elabsTestNoRefineDocksFiltered['distance'][ind]))


            #Write the docks to file with the ligand efficiency as an attribute


            w = Chem.SDWriter(f'{self.storeLoc}/elabsTestNoRefine_Docked_%s.sdf'%index)
            for index in range(len(self.mols_filt)):
                #self.mols_filt[index]
                self.mols_filt[index].SetProp('STRIFE_LigEff_Score', str(self.mols_scores[index]/self.mols_filt[index].GetNumHeavyAtoms()))
                w.write(self.mols_filt[index])
            w.close()
            #Compute ligand Efficiency
            self.elabsTestNoRefineLigEff = self.docking.ligandEfficiency(self.elabsTestNoRefineDocksFiltered['mols'], self.elabsTestNoRefineFS)


            #Write the docks to file with the ligand efficiency as an attribute



            #Standardise the name of the final df
            self.rankedElaborationsFinal = self.elabsTestNoRefineLigEff
            
            if self.writeFinalElabs:
                self.rankedElaborationsFinal.to_csv(f'{self.storeLoc}/rankedElaborationsFinal.csv')
            

            #filter elaborations to find quasi actives: only select mols with max distance below 4.5A
            #selfhotspotsdf_updated = self.HotspotsDF.drop(self.topHotspot.index).reset_index(drop=True)
            selfhotspotsdf_updated = self.HotspotsDF.drop(self.topHotspot.index)
            if len(self.mols_filt) >= 1:
                return(self.mols_filt, selfhotspotsdf_updated)

    def refinement(self, totalNumElabs = 250, n_cores = None):
        #Use the quasi-actives to generate elaborations using the pharm model
        

        n_cores = self.num_cpu_cores


        #i.e. we've specified a certain number of elaborations to make 
        
        #Iterate over all of the single hotspots, make elabs from their quasi actives, append them together and randomly sample totalNumElabs of them
        self.singlePharmElabs = {}
        self.pharmElabsTest = pd.DataFrame()

        with open(self.frag, 'r') as f:
            fragSmiles = f.read()


        self.singlePharmElabs = pharmElabs()
        self.singlePharmElabs.profileElabs = self.elaborate.makePharmElabsQuasiActives(self.singleQuasiActives, fragSmiles, filterMols = False) #make elaborations using the quasi actives profiles
        self.pharmElabsTest = self.pharmElabsTest.append(self.singlePharmElabs.profileElabs)

        #Now we have all of the molecules sampled from the quasi active profiles
        self.pharmElabsTestSample = self.pharmElabsTest.sample(n = totalNumElabs, random_state = 10) #Sample the number of elaborations we want
        self.pharmElabsTestFilter = self.elaborate.filterGeneratedMols(self.pharmElabsTestSample, n_cores = n_cores)

        #Now prepare for docking 
        self.pharmElabsTestFName = f'{self.storeLoc}/pharmsElabsTestPreDocking.sdf'
        self.pharmElabsCountsIdx, self.pharmElabsCountsSDFIdx = self.docking.getSDFs(self.pharmElabsTestFilter, self.fragCore, self.pharmElabsTestFName) 

        #Do docking
        self.pharmElabsTestDocks, self.pharmElabsTestFS = self.docking.dockLigandsMP(self.pharmElabsTestFName, self.constraintFile, self.cavityLigandFile, self.protein, returnFitnessScore = True, n_processes = n_cores)

        #Compute ligand Efficiency
        self.pharmElabsTestLigEff = self.docking.ligandEfficiency(self.pharmElabsTestDocks, self.pharmElabsTestFS)

        #Write the docks to file with the ligand efficiency as an attribute
        self.pharmElabsDockedFName = f'{self.storeLoc}/pharmsElabsTestDocked.sdf'
        w = Chem.SDWriter(self.pharmElabsDockedFName)
        
        for idx, m in enumerate(self.pharmElabsTestDocks):
            m.SetProp('STRIFE_LigEff_Score', str(self.pharmElabsTestFS[idx]/m.GetNumHeavyAtoms()))
            w.write(m)

        w.close()

        #Standardise the name of the final df
        self.rankedElaborationsFinal = self.pharmElabsTestLigEff

        if self.writeFinalElabs:
            self.rankedElaborationsFinal.to_csv(f'{self.storeLoc}/rankedElaborationsFinal.csv')
       

class HotspotSingle:
    def __init__(self, hotspotInfo, frag, clf, constraintFile, cavityLigandFile, protein):

        self.hotspots = hotspotInfo
        self.frag = frag
        self.clf = clf
        self.constraintFile = constraintFile
        self.cavityLigandFile = cavityLigandFile
        self.protein = protein

    
    def vector_angle(self, x, y):
        #Returns the angle between two numpy arrays
        cos_theta = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

        return np.arccos(cos_theta)

    def vector_distance(self, x, y):
        #Returns the distance between two numpy arrays
        diff = np.subtract(x,y)
        return np.linalg.norm(diff)


    def determineProfiles(self):
        #Essentially we just want to determine how many atoms we want to add,
        #in the case where we have an aromatic and the case where we don't

        #In the case where we don't have any aromatics, we just use the predicted atom count (which is the path - 1)

        #The case with aromatics is more difficult but we set the elab length as the combined distance plus 2-3 atoms
        #To allow for different placements of the aromatic ring

        #Fix donor and acceptor Counts
        acceptorCount = 0
        donorCount = 0


        if self.hotspots['type'] == 'Donor':
            donorCount += 1
        elif self.hotspots['type'] == 'Acceptor':
            acceptorCount += 1

        elabLengths = []
        profiles = []

        ###Profile Without Aromatic

        #Obtain number of atoms needed to get to the pharmacophoric point:
        dfForModelPrediction = pd.DataFrame({0:[0], 1:[0], 2:[1], 'dist': [self.hotspots['distFromExit']], 'ang': [self.hotspots['angFromExit']], 'aromaticEnRoute':[0]})
        predPathLength = [int(s) for s in self.clf.predict(dfForModelPrediction)][0]

        elabLengths.append(predPathLength - 1) #-1 because we predict path lengths, not the number of atoms required
        profiles.append([0, 0, acceptorCount, donorCount, 0])


        ###Profile With Aromatic
        dfForModelPredictionAro = pd.DataFrame({0:[0], 1:[0], 2:[1], 'dist': [self.hotspots['distFromExit']], 'ang': [self.hotspots['angFromExit']], 'aromaticEnRoute':[1]})
        predPathLengthAro = [int(s) for s in self.clf.predict(dfForModelPredictionAro)][0]

        elabLengths.append(predPathLengthAro + 1) #Predicted atom count +2
        elabLengths.append(predPathLengthAro + 2) #Predicted atom count +3
        elabLengths.append(predPathLengthAro + 3) #Predicted atom count +4

        profiles.append([0, 0, acceptorCount, donorCount, 1])
        profiles.append([0, 0, acceptorCount, donorCount, 1])
        profiles.append([0, 0, acceptorCount, donorCount, 1])

        return elabLengths, profiles
    
    
    
class HotspotMulti:

    def __init__(self, hotspotInfo, frag, clf, constraintFile, cavityLigandFile, protein):

        self.hotspots = hotspotInfo
        self.frag = frag
        self.clf = clf
        self.constraintFile = constraintFile
        self.cavityLigandFile = cavityLigandFile
        self.protein = protein

    def vector_angle(self, x, y):
        #Returns the angle between two numpy arrays
        cos_theta = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

        return np.arccos(cos_theta)

    def vector_distance(self, x, y):
        #Returns the distance between two numpy arrays
        diff = np.subtract(x,y)
        return np.linalg.norm(diff)


    def determineProfiles(self):
        #The Heuristic for determining the number of atoms we want is based on the
        #distance from the exit point to the furthest atom away, which has index 0
        
        #To account for side chains and aromatic groups, we specify a variety of different elaboration lengths:
            
            #For no aromatic - The predicted distance to the furthest pharmacophoric point plus 1 atom and 2 atoms
            #For aromatic - The predicted distance to the furthest pharmacophoric point plus 2-5 atoms



        #Set donor and acceptor Counts
        acceptorCount = 0
        donorCount = 0


        for k in self.hotspots.keys():
            if self.hotspots[k]['type'] == 'Donor':
                donorCount += 1
            elif self.hotspots[k]['type'] == 'Acceptor':
                acceptorCount += 1

        elabLengths = []
        profiles = []

        ###Profile Without Aromatic

        #Obtain number of atoms needed to get to the first hotspot and from there to the second hotspot:

        dfForModelPrediction = pd.DataFrame({0:[0], 1:[0], 2:[1], 'dist': [self.hotspots[0]['distFromExit']], 'ang': [self.hotspots[0]['angFromExit']], 'aromaticEnRoute':[0]})
        predPathLength = [int(s) for s in self.clf.predict(dfForModelPrediction)][0]

        elabLengths.append(predPathLength - 1) #-1 because we predict path lengths, not the number of atoms required
        elabLengths.append(predPathLength)
        elabLengths.append(predPathLength + 1)
        
        profiles.append([0, 0, acceptorCount, donorCount, 0])
        profiles.append([0, 0, acceptorCount, donorCount, 0])
        profiles.append([0, 0, acceptorCount, donorCount, 0])

        
        ###Profile With Aromatic
        dfForModelPredictionAro = pd.DataFrame({0:[0], 1:[0], 2:[1], 'dist': [self.hotspots[0]['distFromExit']], 'ang': [self.hotspots[0]['angFromExit']], 'aromaticEnRoute':[1]})
        predPathLengthAro = [int(s) for s in self.clf.predict(dfForModelPredictionAro)][0]

        elabLengths.append(predPathLengthAro + 1) #Predicted atom count +2
        elabLengths.append(predPathLengthAro + 2) #Predicted atom count +3
        elabLengths.append(predPathLengthAro + 3) #Predicted atom count + 4
        elabLengths.append(predPathLengthAro + 4) #Predicted atom count + 5


        profiles.append([0, 0, acceptorCount, donorCount, 1])
        profiles.append([0, 0, acceptorCount, donorCount, 1])
        profiles.append([0, 0, acceptorCount, donorCount, 1])
        profiles.append([0, 0, acceptorCount, donorCount, 1])


        return elabLengths, profiles

        


class pharmElabs:
    #Class to store things related to making elaborations on the quasi actives
    #We just initialise it as an empty class
    pass
    

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--fragment_sdf', '-f', type = str, required = True,
                        help = 'Location of fragment SDF. Can be an SDF file of a larger ligand for which the fragment is a substructure')
    parser.add_argument('--fragment_smiles', '-s', type = str, default = None,
                        help = 'Location of file containing fragment smiles string. The exit vector should be denoted by a dummy atom. Either you must specify an exit_vector_index or provide a fragment_smiles.')
    parser.add_argument('--exit_vector_idx', '-v', type = int, default = None,
                        help = 'The atomic index of the atom you wish to use as an exit vector. Either you must specify an exit_vector_index or provide a fragment_smiles. If you provide an exit_vector_index, then STRIFE can only use the molecule provided in fragment_sdf to make elaborations, whereas you can use fragment_smiles to make elaborations to substructures of the molecule provided in fragment_sdf.'  )
    
    
    parser.add_argument('--protein', '-p', type = str, required = True,
                        help = 'Location of protein pdb file (should have already been protonated)')
    parser.add_argument('--output_directory', '-o', type = str, default = '.', 
                        help = 'Directory to store output (default = current directory)')

    parser.add_argument('--load_specified_pharms', '-m', type = str, default = None,
                        help = 'Use pharmacophores that have been manually specfied instead of ones derived from FHMs. If True, the output_directory should contain at least one of donorHotspot.sdf or acceptorHotspot.sdf')    

    parser.add_argument('--path_length_model', type = str, default = 'models/pathLengthPred_saved.pickle', 
                        help = 'Location of saved SVM for predicting path distances')
    

    parser.add_argument('--number_elaborations', '-n', type = int, default = 250,
            help = 'Final number of elaborations for the model to generate. Default: %(default)s')
    parser.add_argument('--number_elaborations_exploration', '-e', type = int, default = 250,
            help = 'Number of elaborations to make per pharmacophoric point in the exploration phase. Default: %(default)s')

    parser.add_argument('--write_elaborations_dataset', '-w', action = "store_true", 
            help='Save the DataFrame containing the final elaborations generated by STRIFE as rankedElaborationsFinal.csv')

    parser.add_argument('--num_cpu_cores', '-cpu', type = int, default = -1, 
            help='Number of CPU cores to use for docking and other computations. Specifiying -1 will use all available cores')
    parser.add_argument('--iter_type', '-it', type=str,default = 'score')
    parser.add_argument('--cluster', type=str,default = False)
    #TODO
    #parser.add_argument('--compute_hotspot_distance', action = "store_true",
            #help='Optional flag which will compute the distance of ligand pharmacophores to the nearest pharmacophoric point and save as a molecule property')

    arguments = parser.parse_args()
    fragments = Chem.SDMolSupplier(arguments.fragment_sdf)
    smiles = pd.read_csv(arguments.fragment_smiles, header=None)
    output_stem = arguments.output_directory
   # for indx in [0]:
    for indx in range(len(fragments)):
        indx = indx + 4

        smi = smiles[0][indx]
        w = Chem.SDWriter(f'{output_stem}/frag.sdf')
        w.write(fragments[indx])
        w.close()
        arguments.fragment_sdf = f'{output_stem}/frag.sdf'
        arguments.fragment_smiles = str(smi)
        arguments.output_directory = f'{output_stem}/frag{indx}'

        #Define STRIFE model
        STRIFE_model = STRIFE(arguments)
        
        #Set it running

        STRIFE_model.run(arguments)
        
        #Pickle the STRIFE Class and SAVE

        run_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        with open(f'{arguments.output_directory}/STRIFE_{run_id}.pickle', 'wb') as f:
            pickle.dump(STRIFE_model, f)

    

    
    
