#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

# External
import random as rnd
import numpy as np
from skimage import morphology
from skimage import transform
from skimage import filters

from collections import OrderedDict

#-----------------------------------------------------------------------------#
#--------------- Functions that create the diferent volume shapes ------------#
#-----------------------------------------------------------------------------#

def fun_cilindro(tam_base):
    aux = morphology.disk(tam_base[0]/2);
    tam_aux = aux.shape
    Salida_Base = np.zeros((tam_aux[0], tam_aux[1], tam_base[2]))
    # Slice-wise drawing
    for i in range(0,tam_base[2]-1):
        Salida_Base[:,:,i] = morphology.disk(tam_base[0]/2)
    return Salida_Base

def fun_estrella_cil_vol(tam_base):
    aux = morphology.star(tam_base[0]/3)
    tam_aux = aux.shape
    Salida_Base = np.zeros((tam_aux[0], tam_aux[1], tam_base[2]))
    # Slice-wise drawing
    for i in range(0,tam_base[2]-1):
        Salida_Base[:,:,i] = morphology.star(tam_base[0]/3)
    return Salida_Base

def fun_estrella_rot_vol(tam_base):
    aux = morphology.star(tam_base/3)
    tam_aux = aux.shape
    Salida_Base = np.zeros((tam_aux[0], tam_aux[0], tam_aux[0]))
    # I create a star in the center of the volume, wich must be a cube
    Salida_Base[:,:,tam_base/2] = morphology.star(tam_base/3)
    Salida_aux = Salida_Base;
    # Now I rotate it over its X axis
    for i in range(0,360,1):
        Salida_Base = Salida_Base + fun_rotar_vol(Salida_aux, 0, i,0)
    # Saturate output...
    Salida_Base[np.where(Salida_Base > 1)] = 1
    return Salida_Base

#-----------------------------------------------------------------------------#
#--------------- Volume transformation ---------------------------------------#
#-----------------------------------------------------------------------------#

def fun_rotar_vol(Vol_entrada, eje, angulo, expandir):    
    # Rotating volume shape
    tam_base = Vol_entrada.shape
    # Output volume generation
    if expandir == 1:
        # If the volume is to be expanded due to its rotation, I check the new size
        if eje == 0:
            aux_rot = transform.rotate(Vol_entrada[1,:,:],angulo,1)
            tam_aux = aux_rot.shape
            Salida_Base = np.zeros((tam_base[0], tam_aux[0], tam_aux[1]))
        if eje == 1:
            aux_rot = transform.rotate(Vol_entrada[:,1,:],angulo,1)
            tam_aux = aux_rot.shape
            Salida_Base = np.zeros((tam_aux[0], tam_base[1], tam_aux[1]))
        if eje == 2:
            aux_rot = transform.rotate(Vol_entrada[:,:,1],angulo,1)
            tam_aux = aux_rot.shape
            Salida_Base = np.zeros((tam_aux[0], tam_aux[1], tam_base[2]))
    else:
        Salida_Base = np.zeros(tam_base)
    # Slice-wise rotation
    for i in range(0,tam_base[eje]-1):
        if eje == 0:
            Salida_Base[i,:,:] = transform.rotate(Vol_entrada[i,:,:],angulo,expandir, preserve_range=1)
        if eje == 1:
            Salida_Base[:,i,:] = transform.rotate(Vol_entrada[:,i,:],angulo,expandir, preserve_range=1)
        if eje == 2:
            Salida_Base[:,:,i] = transform.rotate(Vol_entrada[:,:,i],angulo,expandir, preserve_range=1)
    return Salida_Base


#-----------------------------------------------------------------------------#
#--------------- Random Volume generation ------------------------------------#
#-----------------------------------------------------------------------------#

def fun_generar_vol_random(tipo_forma, tam_limite, transp_forma ):
    # Base volume generation
    if (tipo_forma == 1):
        # Random size
        tam_aux = rnd.randint(tam_limite[0],tam_limite[1])
        # Shape creation
        vol_forma = morphology.cube(tam_aux)

    elif(tipo_forma == 2):
        # Random size
        tam_aux = rnd.randint(tam_limite[0],tam_limite[1])
        # Shape creation
        vol_forma = morphology.ball(tam_aux)
        
    elif(tipo_forma == 3):
        # Random size
        tam_forma = (rnd.randint(tam_limite[0],tam_limite[1]) , 
                     rnd.randint(tam_limite[0],tam_limite[1]) , 
                     rnd.randint(tam_limite[0]*2,tam_limite[1]*2))   
        # Shape creation
        vol_forma = fun_cilindro(tam_forma)

    elif(tipo_forma == 4):
        # Random size
        tam_aux = rnd.randint(tam_limite[0],tam_limite[1])
        # Shape creation
        vol_forma = morphology.octahedron(tam_aux)

    elif(tipo_forma == 5):
        # Random size
        tam_forma = (rnd.randint(tam_limite[0],tam_limite[1]) , 
                     rnd.randint(tam_limite[0],tam_limite[1]) , 
                     rnd.randint(tam_limite[0]*2,tam_limite[1]*2))   
        # Shape creation
        vol_forma = fun_estrella_cil_vol(tam_forma)

    elif(tipo_forma == 6):
        # Random size
        tam_aux = rnd.randint(tam_limite[0],tam_limite[1])
        # Shape creation
        vol_forma = fun_estrella_rot_vol(tam_aux)
        
    # Shape label population
    label_forma = np.multiply(vol_forma,tipo_forma)
    # Transparency
    vol_forma = np.multiply(vol_forma,transp_forma)
    tam_forma = vol_forma.shape
        
    return (tam_forma, vol_forma, label_forma)



def fun_poblar_vol_random(vol_poblar, cantidad_elementos, margenes, tam_limite, transparencia_limite, train_single_class):
    
    if (train_single_class != 0):
        Numero_de_canales_verdad = 3
    else:
        Numero_de_canales_verdad = 6+1
        
    tam_base = vol_poblar.shape
    verdad_base = np.zeros((tam_base[0], tam_base[1], tam_base[2], Numero_de_canales_verdad))
       
    for i in range(0,cantidad_elementos):
        
        # Random shape, according to following key:
        # -- Cube
        # -- Ball
        # -- Cilinder
        # -- Octahedro
        # -- Extruded Star
        # -- Revolution Star
        Numero_de_formas = 6
        tipo_forma = rnd.randint(1,Numero_de_formas)
        
        # If this function is called from a network training only one shape type, then 
        # I assure that at least that shape appears one time.
        if train_single_class != 0 and i == 0:
            tipo_forma = train_single_class
            
        # Random transparency
        transp_forma = rnd.uniform(transparencia_limite[0], transparencia_limite[1])
        
        # Shape generation
        (tam_forma, vol_forma, label_forma) = fun_generar_vol_random(tipo_forma, tam_limite, transp_forma )

        # Random Rotation
        eje_rot = rnd.randint(0,2)
        grados_rot = rnd.uniform(-90,90)
        vol_forma_aux = fun_rotar_vol(vol_forma, eje_rot, grados_rot, 1)
        label_forma_aux = fun_rotar_vol(label_forma, eje_rot, grados_rot, 1)
        tam_forma_aux = vol_forma_aux.shape
        
        # Chek bounds after rotations, and abort rotation in case it violates any bound
        if ( (tam_base[0] - margenes[1] - (tam_forma_aux[0]/2)) - (margenes[0]+(tam_forma_aux[0]/2)) ) >= 1 and ( (tam_base[1] - margenes[3] - (tam_forma_aux[1]/2)) - (margenes[2]+(tam_forma_aux[1]/2)) ) >= 1 and ( (tam_base[2] - margenes[5] - (tam_forma_aux[2]/2)) - (margenes[4]+(tam_forma_aux[2]/2)) ) >= 1:
            vol_forma = vol_forma_aux
            label_forma = label_forma_aux
            tam_forma = tam_forma_aux

            
        # After creating the volume, I place it at a random position inside the base volume
        
        # Random position
        centro_forma = (rnd.randint(margenes[0]+(tam_forma[0]/2),tam_base[0] - margenes[1] - (tam_forma[0]/2)) , 
                        rnd.randint(margenes[2]+(tam_forma[1]/2),tam_base[1] - margenes[3] - (tam_forma[1]/2)) , 
                        rnd.randint(margenes[4]+(tam_forma[2]/2),tam_base[2] - margenes[5] - (tam_forma[2]/2)) )
        
        # Corner of the shape in base volume coordinates
        centro_forma = (centro_forma[0]-(tam_forma[0]/2) , 
                        centro_forma[1]-(tam_forma[1]/2) , 
                        centro_forma[2]-(tam_forma[2]/2) )
        
        # Addition to base volume
        vol_poblar[centro_forma[0]:centro_forma[0]+tam_forma[0],
                   centro_forma[1]:centro_forma[1]+tam_forma[1],
                   centro_forma[2]:centro_forma[2]+tam_forma[2]]  =  vol_forma +  vol_poblar[centro_forma[0]:centro_forma[0]+tam_forma[0], centro_forma[1]:centro_forma[1]+tam_forma[1],              centro_forma[2]:centro_forma[2]+tam_forma[2]]     

        # Background label fill
        verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0],
                                   centro_forma[1]:centro_forma[1]+tam_forma[1],
                                   centro_forma[2]:centro_forma[2]+tam_forma[2],
                               0]  = label_forma +  verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0], centro_forma[1]:centro_forma[1]+tam_forma[1], centro_forma[2]:centro_forma[2]+tam_forma[2], 0]
    
        # Fill class label according to type (single or multiclass)
        if (train_single_class == 0):
            verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0],
                           centro_forma[1]:centro_forma[1]+tam_forma[1],
                           centro_forma[2]:centro_forma[2]+tam_forma[2],
                       tipo_forma]  =  label_forma +  verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0], centro_forma[1]:centro_forma[1]+tam_forma[1], centro_forma[2]:centro_forma[2]+tam_forma[2], tipo_forma]
    
        # If it is "single class" I fill the "others" class with all non target shapes
        elif (train_single_class == label_forma.max()):
            verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0],
                           centro_forma[1]:centro_forma[1]+tam_forma[1],
                           centro_forma[2]:centro_forma[2]+tam_forma[2],
                       1]  = label_forma +  verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0], centro_forma[1]:centro_forma[1]+tam_forma[1], centro_forma[2]:centro_forma[2]+tam_forma[2], 0]
        elif (train_single_class != label_forma.max()):
            verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0],
                           centro_forma[1]:centro_forma[1]+tam_forma[1],
                           centro_forma[2]:centro_forma[2]+tam_forma[2],
                       2]  = label_forma +  verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0], centro_forma[1]:centro_forma[1]+tam_forma[1], centro_forma[2]:centro_forma[2]+tam_forma[2], 0]
        
        
    # Output clipping (safety)
    verdad_base = np.clip(verdad_base, 0, 1)
    # I invert the background channel to generate a network fiendly class.
    verdad_base[:,:,:,0] = 1 - verdad_base[:,:,:,0]
    # If it is single class the overlappling zones are deleted from "other" class
    if (train_single_class != 0):
        dims = verdad_base.shape
        for i in range(0,dims[0]):
            for j in range(0,dims[1]):
                for k in range(0,dims[2]):
                    if verdad_base[i,j,k,2] == verdad_base[i,j,k,1]:
                            verdad_base[i,j,k,2] = 0


    return (vol_poblar, verdad_base)


def fun_poblar_vol_each(vol_poblar, margenes, tam_limite, transparencia_limite, noise_max, gauss_sigma, repetitions, train_single_class):
    
    # This function generates a volume with one element of each class, used to show performance
    
    if (train_single_class != 0):
        Numero_de_canales_verdad = 3
    else:
        Numero_de_canales_verdad = 6+1
        
    tam_base = vol_poblar.shape
    verdad_base = np.zeros((tam_base[0], tam_base[1], tam_base[2], Numero_de_canales_verdad))
    
    # Shapes to be used:
    # -- Cube
    # -- Ball
    # -- Cilinder
    # -- Octahedro
    # -- Extruded Star
    # -- Revolution Star
    Numero_de_formas = 6

    for rep in range(0,repetitions):
        for i in range(0,Numero_de_formas):


            tipo_forma = i+1

            # Random transparency
            transp_forma = rnd.uniform(transparencia_limite[0], transparencia_limite[1])

            # Shape generation
            (tam_forma, vol_forma, label_forma) = fun_generar_vol_random(tipo_forma, tam_limite, transp_forma )

            # Random Rotation
            eje_rot = rnd.randint(0,2)
            grados_rot = rnd.uniform(-90,90)
            vol_forma_aux = fun_rotar_vol(vol_forma, eje_rot, grados_rot, 1)
            label_forma_aux = fun_rotar_vol(label_forma, eje_rot, grados_rot, 1)
            tam_forma_aux = vol_forma_aux.shape

            # Chek bounds after rotations, and abort rotation in case it violates any bound
            if ( (tam_base[0] - margenes[1] - (tam_forma_aux[0]/2)) - (margenes[0]+(tam_forma_aux[0]/2)) ) >= 1 and ( (tam_base[1] - margenes[3] - (tam_forma_aux[1]/2)) - (margenes[2]+(tam_forma_aux[1]/2)) ) >= 1 and ( (tam_base[2] - margenes[5] - (tam_forma_aux[2]/2)) - (margenes[4]+(tam_forma_aux[2]/2)) ) >= 1:
                vol_forma = vol_forma_aux
                label_forma = label_forma_aux
                tam_forma = tam_forma_aux


            # After creating the volume, I place it at a random position inside the base volume

            # Random position
            centro_forma = (rnd.randint(margenes[0]+(tam_forma[0]/2),tam_base[0] - margenes[1] - (tam_forma[0]/2)) , 
                            rnd.randint(margenes[2]+(tam_forma[1]/2),tam_base[1] - margenes[3] - (tam_forma[1]/2)) , 
                            rnd.randint(margenes[4]+(tam_forma[2]/2),tam_base[2] - margenes[5] - (tam_forma[2]/2)) )

            # Corner of the shape in base volume coordinates
            centro_forma = (centro_forma[0]-(tam_forma[0]/2) , 
                            centro_forma[1]-(tam_forma[1]/2) , 
                            centro_forma[2]-(tam_forma[2]/2) )

            # Addition to base volume
            vol_poblar[centro_forma[0]:centro_forma[0]+tam_forma[0],
                       centro_forma[1]:centro_forma[1]+tam_forma[1],
                       centro_forma[2]:centro_forma[2]+tam_forma[2]]  =  vol_forma +  vol_poblar[centro_forma[0]:centro_forma[0]+tam_forma[0], centro_forma[1]:centro_forma[1]+tam_forma[1],              centro_forma[2]:centro_forma[2]+tam_forma[2]]     

            # Background label fill
            verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0],
                                       centro_forma[1]:centro_forma[1]+tam_forma[1],
                                       centro_forma[2]:centro_forma[2]+tam_forma[2],
                                   0]  = label_forma +  verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0], centro_forma[1]:centro_forma[1]+tam_forma[1], centro_forma[2]:centro_forma[2]+tam_forma[2], 0]

            # Fill class label according to type (single or multiclass)
            if (train_single_class == 0):
                verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0],
                               centro_forma[1]:centro_forma[1]+tam_forma[1],
                               centro_forma[2]:centro_forma[2]+tam_forma[2],
                           tipo_forma]  =  label_forma +  verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0], centro_forma[1]:centro_forma[1]+tam_forma[1], centro_forma[2]:centro_forma[2]+tam_forma[2], tipo_forma]

            # If it is "single class" I fill the "others" class with all non target shapes
            elif (train_single_class == label_forma.max()):
                verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0],
                               centro_forma[1]:centro_forma[1]+tam_forma[1],
                               centro_forma[2]:centro_forma[2]+tam_forma[2],
                           1]  = label_forma +  verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0], centro_forma[1]:centro_forma[1]+tam_forma[1], centro_forma[2]:centro_forma[2]+tam_forma[2], 0]
            elif (train_single_class != label_forma.max()):
                verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0],
                               centro_forma[1]:centro_forma[1]+tam_forma[1],
                               centro_forma[2]:centro_forma[2]+tam_forma[2],
                           2]  = label_forma +  verdad_base[centro_forma[0]:centro_forma[0]+tam_forma[0], centro_forma[1]:centro_forma[1]+tam_forma[1], centro_forma[2]:centro_forma[2]+tam_forma[2], 0]


        # Output clipping (safety)
        verdad_base = np.clip(verdad_base, 0, 1)
        # I invert the background channel to generate a network fiendly class.
        verdad_base[:,:,:,0] = 1 - verdad_base[:,:,:,0]
        # If it is single class the overlappling zones are deleted from "other" class
        if (train_single_class != 0):
            dims = verdad_base.shape
            for i in range(0,dims[0]):
                for j in range(0,dims[1]):
                    for k in range(0,dims[2]):
                        if verdad_base[i,j,k,2] == verdad_base[i,j,k,1]:
                                verdad_base[i,j,k,2] = 0


    # Noise addition and filtering
    vol_poblar = filters.gaussian(vol_poblar, sigma = gauss_sigma, preserve_range = True)
    vol_noise = np.random.rand(tam_base[0], tam_base[1], tam_base[2])
    vol_noise = np.multiply(vol_noise, noise_max)
    vol_poblar += vol_noise
    
    return (vol_poblar, verdad_base)
    

    
#-----------------------------------------------------------------------------#
#--------------- Multiple Random Volume generation ---------------------------#
#-----------------------------------------------------------------------------#
    
def fun_gen_batch(cant_cubos, tam_vol_poblar, noise_max, cantidad_elementos, margenes, tam_limite, transparencia_limite, gauss_sigma, train_single_class):
    
    if (train_single_class != 0):
        # Backgound - Target class - Others Class
        Numero_de_canales_verdad = 3
    else:
        # One clase per shape + Background 
        Numero_de_canales_verdad = 6+1
    
    # Output allocation
    batch_vols = np.zeros((cant_cubos, tam_vol_poblar[0], tam_vol_poblar[1], tam_vol_poblar[2]))
    batch_labels = np.zeros((cant_cubos, tam_vol_poblar[0], tam_vol_poblar[1], tam_vol_poblar[2], Numero_de_canales_verdad))
    
    # Generation loop
    for i in range(0,cant_cubos):
        
        # Volume to populate
        vol_poblar = np.zeros((tam_vol_poblar[0], tam_vol_poblar[1], tam_vol_poblar[2]))

        # Volume population
        (batch_vols[i,:,:,:], batch_labels[i,:,:,:,:]) = fun_poblar_vol_random(vol_poblar, cantidad_elementos, margenes, tam_limite, transparencia_limite, train_single_class)
        
        # Noise addition and filtering
        batch_vols[i,:,:,:] = filters.gaussian(batch_vols[i,:,:,:], sigma = gauss_sigma, preserve_range = True)
        vol_noise = np.random.rand(tam_vol_poblar[0], tam_vol_poblar[1], tam_vol_poblar[2])
        vol_noise = np.multiply(vol_noise, noise_max)
        batch_vols[i,:,:,:] += vol_noise
        
        
    return (batch_vols, batch_labels)

    
    
    
    
    
    
