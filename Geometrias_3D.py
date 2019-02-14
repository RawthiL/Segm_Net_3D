#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

# External
import random as rnd
import numpy as np
from skimage import morphology
from skimage import transform
from skimage import filters

from collections import OrderedDict



import multiprocessing
import math
from scipy import ndimage
from skimage import measure

from joblib import Parallel, delayed
import multiprocessing

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





#-----------------------------------------------------------------------------#
#--------------- Random Volume Transformation --------------------------------#
#-----------------------------------------------------------------------------#


# http://nbviewer.jupyter.org/gist/lhk/f05ee20b5a826e4c8b9bb3e528348688
# http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

def unit_vector(data, axis=None, out=None):
   
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix(angle, direction, point=None):
    
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M



def rotate_mesh(input_mesh, size_fov, angle, direction):
    
    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz=np.vstack([input_mesh[0].reshape(-1)-float(size_fov[0])/2,     # x coordinate, centered
    input_mesh[1].reshape(-1)-float(size_fov[1])/2,     # y coordinate, centered
    input_mesh[2].reshape(-1)-float(size_fov[2])/2,     # z coordinate, centered
    np.ones((size_fov[0],size_fov[1],size_fov[2])).reshape(-1)])    # 1 for homogeneous coordinates
    
    
    # create transformation matrix
    mat=rotation_matrix(angle,direction)
    
    
    # apply transformation
    transformed_xyz=np.dot(mat, xyz)
    
    # extract coordinates, don't use transformed_xyz[3,:] that's the homogeneous coordinate, always 1
    x=transformed_xyz[0,:]+float(size_fov[0])/2
    y=transformed_xyz[1,:]+float(size_fov[1])/2
    z=transformed_xyz[2,:]+float(size_fov[2])/2
    
    x=x.reshape((size_fov[0],size_fov[1],size_fov[2]))
    y=y.reshape((size_fov[0],size_fov[1],size_fov[2]))
    z=z.reshape((size_fov[0],size_fov[1],size_fov[2]))

    return [x,y,z]



def ripple_mesh(input_mesh, size_fov, max_ripple, wave_peak, wave_periode):
    
    # Create a volumetric rippled fov

    # Aux Volume
    rip_vol = np.zeros((size_fov[0], size_fov[1], size_fov[2]))

    # Create an uniform grid
    ax = OrderedDict()
    ax[0] = np.arange(size_fov[0]).astype(np.float32)
    ax[1] = np.arange(size_fov[1]).astype(np.float32)
    ax[2] = np.arange(size_fov[2]).astype(np.float32)

    # First implement the sine across all axis
    aux_sins = OrderedDict()
    for i in range(0,3):
        aux_sins[i] = np.multiply(wave_peak[i],np.sin(ax[i]/wave_periode[i]))

    # X-axis
    x_xpand = np.expand_dims(np.expand_dims(aux_sins[0], axis=(1)), axis=(1))
    x_rip_vol = np.tile(x_xpand, (1,240,155))
    # Y-axis
    y_xpand = np.expand_dims(np.expand_dims(aux_sins[1], axis=(1)), axis=(0))
    y_rip_vol = np.tile(y_xpand, (240,1,155))
    # X-axis
    z_xpand = np.expand_dims(np.expand_dims(aux_sins[2], axis=(0)), axis=(0))
    z_rip_vol = np.tile(z_xpand, (240,240,1))

    # Compose
    rip_vol = np.multiply(np.multiply(x_xpand,y_xpand),z_xpand)

    # Re-scale
    rip_vol = np.divide(rip_vol,rip_vol.max())
    rip_vol = np.multiply(rip_vol,max_ripple)

    # apply
    input_mesh[0] += rip_vol
    input_mesh[1] += rip_vol
    input_mesh[2] += rip_vol
    
    x=input_mesh[0].reshape((size_fov[0],size_fov[1],size_fov[2]))
    y=input_mesh[1].reshape((size_fov[0],size_fov[1],size_fov[2]))
    z=input_mesh[2].reshape((size_fov[0],size_fov[1],size_fov[2]))
    
    return [x,y,z]
    
    


def random_ripple_and_rotation(input_vol, max_ripple_range, wave_periode_range, spline_order, num_cores=multiprocessing.cpu_count(), multi_channel=False):
    
    
    # Random max ripple
    ripple_val = rnd.uniform(max_ripple_range[0], max_ripple_range[1])
    
    # Random wave peaks
    wave_peak = np.zeros((3))
    wave_peak[0] = rnd.uniform(0,1)
    wave_peak[1] = rnd.uniform(0,1)
    wave_peak[2] = rnd.uniform(0,1)
    
    # Random wave periods
    wave_period = np.zeros((3))
    wave_period[0] = rnd.uniform(max_ripple_range[0], max_ripple_range[1])
    wave_period[1] = rnd.uniform(max_ripple_range[0], max_ripple_range[1])
    wave_period[2] = rnd.uniform(max_ripple_range[0], max_ripple_range[1])
    
    # Random rotate angle
    rot_angle = rnd.uniform(-np.pi/4,np.pi/4)
    
    # Random rotate direction (the unity norm is handled by "unit_vector" )
    rot_dir = np.zeros((3))
    rot_dir[0] = rnd.uniform(max_ripple_range[0], max_ripple_range[1])
    rot_dir[1] = rnd.uniform(max_ripple_range[0], max_ripple_range[1])
    rot_dir[2] = rnd.uniform(max_ripple_range[0], max_ripple_range[1])
    
    # Get input shape
    size_fov = input_vol.shape
    if multi_channel:
        channels = size_fov[0]
        size_fov = size_fov[1:4]
    
    # Create an uniform grid
    ax_x = np.arange(size_fov[0]).astype(np.float32)
    ax_y = np.arange(size_fov[1]).astype(np.float32)
    ax_z = np.arange(size_fov[2]).astype(np.float32)
    
    aux_mesh = np.meshgrid(ax_x,ax_y,ax_z)
    
    # Rotate the mesh
    aux_mesh = rotate_mesh(aux_mesh, size_fov, rot_angle, rot_dir)

    # Ripple the mesh
    aux_mesh = ripple_mesh(aux_mesh, size_fov, ripple_val, wave_peak, wave_period)

    # the coordinate system seems to be strange, it has to be ordered like this
    aux_mesh=[aux_mesh[1],aux_mesh[0],aux_mesh[2]]
    
    # Re-sample
    if multi_channel:
        # To make this faster we paralellize the operation
        new_vol = np.zeros((channels,size_fov[0],size_fov[1],size_fov[2]))
        par_results =  Parallel(n_jobs=num_cores)(delayed(fun_multiple_map_coordinates)
                                              (size_fov=size_fov,
                                               input_vol=input_vol,
                                               aux_mesh = aux_mesh,
                                               spline_order = spline_order,
                                               num_vol = i) for i in range(0, channels))
        # Collect results
        for idx_ch in range(channels):
            new_vol[idx_ch,:,:,:] = par_results[idx_ch]
        
    else:
        new_vol=scipy.ndimage.map_coordinates(input_vol,aux_mesh, order=spline_order)
    
    return new_vol

    

def fun_multiple_map_coordinates(size_fov, input_vol, aux_mesh, spline_order, num_vol):
    # https://bic-berkeley.github.io/psych-214-fall-2016/map_coordinates.html
    new_vol = np.zeros((size_fov[0],size_fov[1],size_fov[2]))
    new_vol= ndimage.map_coordinates(input_vol[num_vol,:,:,:],aux_mesh, order=spline_order)
    return new_vol
    
    
    
    
