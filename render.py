# Import the library mitsuba using the alias 'mi'
import mitsuba as mi

# Set the variant of the renderer
mi.set_variant('cuda_ad_rgb')

# Import other libraries
from mitsuba import ScalarTransform4f as T
import matplotlib.pyplot as plt
import drjit as dr
from tqdm import tqdm
from utils import *
import os
import argparse

# Define the parser
parser = argparse.ArgumentParser(description='Rendering pipeline of the analytic dataset.')

""" Selec the color type for the images:
        0 fixed color
        (1,2,3) subset of a random color for each image
"""
parser.add_argument('-c', '--color_type', type=int, default=0)   

# If color_type is setted to 0, this color will be used for all images in the dataset
parser.add_argument('-f', '--fixed_color', nargs=3, type=float, default=[0.25, 0.25, 0.25]) 

# Path to xml files
parser.add_argument('-i', '--input_path', type=str, default="xmls/")

# Output path
parser.add_argument('-o', '--output_path', type=str, required=True) 

# Number of samples for the roughness and specular parameters
parser.add_argument('-r', '--n_roughness', type=int, default=5) 
parser.add_argument('-s', '--n_specular', type=int, default=10)   

# Max angle of random ratation for geometries
parser.add_argument('-a', '--max_angle', type=float, default=15.0)   

args = parser.parse_args()

init_index = args.color_type
xml_files = os.listdir(args.input_path)
xml_files.sort()
max_angle = args.max_angle
n_specular = args.n_specular
n_roughness = args.n_roughness
fixed_color = args.fixed_color

# Labels for the three kind of colors for each image
color_labels = ['fixed_color', 'random_color_1', 'random_color_2', 'random_color_3']

# Logg of parametres used during rendering for each image
data = open(args.output_path + '_analytic_materials_{}.csv'.format(color_labels[init_index]),'w')
data.write('{}, {}, {}, {}, {}, {} \n'.format('image', 'roughness', 'specular', 'R', 'G', 'B'))

if not os.path.exists(args.output_path + '_renders_{}/'.format(color_labels[init_index])):
    os.mkdir(args.output_path + '_renders_{}/'.format(color_labels[init_index]))

# Rendering pipeline
for id_scene in tqdm(range(len(xml_files))): 

    # Load scene parameters
    file = xml_files[id_scene]
    scene = mi.load_file(args.input_path + file)  

    # Roughness and specular sampling
    roughness_values = np.linspace(0.1, 0.4, num = n_roughness)
    specular_values = np.geomspace(0.2, 4.5, num = n_specular)

    # Get scene parameters
    params = mi.traverse(scene)

    # Get geometry vertex positions
    V = dr.unravel(mi.Point3f, params['OBJMesh.vertex_positions'])

    for r in range(n_roughness):

        # Roughness value update
        roughness_value = roughness_values[r]
        params['bsdf-matpreview.roughness.value'] = roughness_value
        
        for s in range(n_specular):

            # Specular value update
            specular_value = specular_values[s]
            params['bsdf-matpreview.specular'] =  specular_value

            # Geometry random rotation
            angle_x = (random.random() * max_angle * 2) - max_angle
            angle_y = (random.random() * max_angle * 2) - max_angle
            R_X = mi.Transform4f.rotate(axis=[1,0,0], angle=angle_x)
            R_Y = mi.Transform4f.rotate(axis=[0,1,0], angle=angle_y)
            RV = R_Y @ R_X @ V
            params['OBJMesh.vertex_positions'] = dr.ravel(RV)

            if init_index == 0:
                color = fixed_color
            else:
                color =color_hsluv_gauss() 

            # Scene parameters update
            params['bsdf-matpreview.base_color.value'] = color
            params.update()

            # Rendering image
            img = mi.render(scene)
            
            # Save image
            bitmap = mi.Bitmap(img[:,:,:11], channel_names=['color.R', 'color.G', 'color.B'] 
                + ['normal.R', 'normal.G', 'normal.B', 'distance.L', 'object.A', 'object.R', 'object.G', 'object.B'])
            output_name = args.output_path + '_renders_{}/'.format(color_labels[init_index]) + file[:-4] + "[analytic-{:.2f}-{:.2f}-{}]".format(roughness_value, specular_value, color_labels[init_index])
            bitmap.write('{}.exr'.format(output_name))
            data.write('{}.exr, {}, {}, {}, {}, {}\n'.format(output_name, roughness_value, specular_value, color[0], color[1], color[2]))

data.close()
