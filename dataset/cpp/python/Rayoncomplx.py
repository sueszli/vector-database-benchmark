#!/usr/bin/env python
# -*- coding: utf-8 -*-



# Importation des modules standards
import numpy as np
import matplotlib.pyplot as mpl
from scipy import optimize
from operator import mul
from matplotlib.patches import Ellipse
from functools import *

#from functools import *

def ABCD(A,B,C,D):
	'''Fonction toute simple permettant de créer une matrice ABCD avec les indices'''
	return np.matrix(((A,B),(C,D)))

class RayonComplexe:
	'''Objet dont l'init permet de stocker les caractéristiques dun faisceau gaussien'''
	def __init__(self, complx,longueur):
		self.q = complx
		self.l = longueur *10**(-9)
		self.etranglement = np.sqrt((self.l * self.q.imag)/np.pi)
	def rayon(self,dist):
		return (self.q.real+dist) * np.sqrt(1+(self.q.imag/(self.q.real+dist))**2)
	def width(self,dist):
		return self.etranglement * np.sqrt(1+((self.q.real+dist)/self.q.imag)**2)
	def transform(self,compo):
		'''La loi de transformation ou compo est une matrice ABCD'''
		return RayonComplexe(((compo[0,0] * self.q) + compo[0,1])/((compo[1,0] * self.q) + compo[1,1]),self.l*10**9)

def syst4f(f1,f2):
	return ABCD(-float(f2)/f1,0,0,-(float(f1)/f2))

def syst2f(f):
	return ABCD(0,f,-float(1)/f,0)

def space(dist):
	return ABCD(1,dist,0,1)

def lens(f):
	return ABCD(1,0,-float(1)/f,1)

def cheminoptiquerayon(rayoncomplx,system,pos=False,width=False,complx=False, label = []):
	'''Renvoie une liste avec un objet RayonComplexe à la sortie de chaque matrice ABCD. En option: complx renvoie une liste de nmbr complexe, 
	width renvoie une liste de taille et pos permet de fixer quelles matrices seront dans la liste de sortie (permet par exemple de sauter les
	espaces.'''
	if pos == False:
		sigma = range(len(system))
	else:
		sigma = pos
	liste = []
	for i in sigma:
		liste.append(rayoncomplx.transform(reduce(mul, list(reversed(system[0:i+1])), 1)))
	if complx==True:
		for i in range(len(liste)):
			liste[i] = liste[i].q
	if width==True:
		for i in range(len(liste)):
			liste[i] = liste[i].width(0)
	for i in range(len(liste)):
		if i in label and system[i][0,1] != 0:
			if i is not 0:
				print('a la fin de l\'espace vide, '+ str(i) + 'e objet, q = ' + str(liste[i].q))
			else:
				print('a la fin de l\'espace vide, '+ str(i) + 'e objet, q = ' + str(liste[0].q))
		elif i in label:
			print('a la lentille '+str(i)+', nous avoons q = ' + str(liste[i].q))
	print('\n')


	return liste

def plot(fct, deb, fin, offset=1j, div=1000000, label = False):
    """Simple fonction permettant de produire un graphique avec une fonction a un argument en entrée"""
    if isinstance(offset, complex):
    	offset = -deb
    xdiv = np.linspace(deb, fin, div)
    ydiv = fct(xdiv+offset)	
    if label:
    	label = ydiv[-1]
    	for y in range(1,len(ydiv)):
    		if ydiv[y] > ydiv[y-1]:
    			label = ydiv[y]
    			break
    	nround = 5 - int(np.ceil(np.log10(label*10**6)))
    	if nround <= 0:
    		mpl.plot(xdiv,ydiv, label = str(int(label*10**6))+'$\mu m$' )
    	else:
    		mpl.plot(xdiv,ydiv, label = str(round(label*10**6,nround))+'$\mu m$' )
    else:
    	mpl.plot(xdiv,ydiv)

def tracerLentille(pos, hauteur, longtot, hauteurmax):
	l = (longtot/20*hauteur/hauteurmax)**2
	x = [pos-np.sqrt(l)]
	y = [0]
	v = [pos-np.sqrt(l)]
	w = [0]
	for i in np.linspace(-np.sqrt(l),np.sqrt(l),5000):
		y.append(np.sqrt(hauteur**2*(1-i**2/l)))
		x.append(i+pos)
		y.append(0)
		x.append(i+pos)
		w.append(np.sqrt(hauteur**2*(1-i**2/l)))
		v.append(i+pos)
	x.append(pos+np.sqrt(l))
	y.append(0)
	v.append(pos+np.sqrt(l))
	w.append(0)
	mpl.plot(x,y, color = 'lightblue')
	mpl.plot(v,w,'k')

def propagation(rayon,wavelength = 488, systeme = [], affiches = []):
	a = RayonComplexe(rayon,wavelength)
	b = []
	position = [0]
	longtot = 0
	hauteurmax = 0
	assert not isinstance(systeme[0],complex)
	for i in systeme:
		if isinstance(i,complex):
			b.append(lens(i.imag))
			position.append(position[-1])
		else:
			longtot = longtot + i
			b.append(space(i))
			position.append(i+position[-1])

	c = cheminoptiquerayon(a,b, label = affiches)
	for i in c:
		if i.width(0) > hauteurmax:
			hauteurmax = i.width(0)
	for i in range(len(c)):
		if position[i] == position[i+1]:
			tracerLentille(position[i],c[i].width(0), longtot, hauteurmax)
	for i in range(len(c)):
		if position[i] != position[i+1]:
			if i in affiches:
				plot(c[i].width, position[i],position[i+1],-position[i+1], label = True)
			else:
				plot(c[i].width, position[i],position[i+1],-position[i+1])


	mpl.suptitle('Propagation d\'un laser de '+str(wavelength)+' nm dans le microscope, avec q = '+str(rayon))
	mpl.ylabel('Rayon du faisceau (m)')
	mpl.xlabel('Position dans le montage (m)')
	mpl.legend()
	mpl.show()


