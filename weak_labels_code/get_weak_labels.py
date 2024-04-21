import json
import os
from utils import *
from tqdm import tqdm

path = '../image_data'
files = os.listdir(path)
files.sort()

ill_dict = { 
            '1_1_cambridge_2k' : 1,
            '1_2_fish_eagle_hill_2k' : 2,
            '1_3_circus_maximus_1_2k' :3,
            '2_1_small_cathedral_2k_shift_-600' : 4,
            '2_2_art_studio_2k' : 5,
            '2_3_tiber_1_2k_shift_-200' : 6,
            '3_1_auto_service_2k_shift_300'  :7,
            '3_2_chinese_garden_2k_shift_500' : 8,
            '3_3_ninomaru_teien_2k_shift_200' : 9,
            'brown_photostudio_02_2k' : 10, 
            'cabin_2k' : 11, 
            'canary_wharf_2k' : 12,
            'christmas_photo_studio_04_2k' : 13, 
            'cyclorama_hard_light_2k' : 14, 
            'dirt_bike_track_01_2k' : 15,
            'fouriesburg_mountain_cloudy_2k': 16, 
            'hilly_terrain_01_2k' : 17, 
            'st_peters_square_night_2k' : 18
            }

brdf_data = open("dataset_physical_parameters.txt", "r")
brdf_data_lines = brdf_data.readlines()
img2color = {}

for i in brdf_data_lines[1:]:
    params = i[:-1].split(',')
    img2color[params[0]] = params[3:]

brdf_data.close()

img2statistics = {}
is_labels = open("image_statistics_labels.txt", "r")
is_labels_lines = is_labels.readlines()

for l in is_labels_lines:

    params = l[:-1].split(',')
    ill, geom, mat = params[0].split('][')
    label = params[1]
    mat = mat.split(']')
    specular = float(mat[0].split('-')[2])
    rougness = float(mat[0].split('-')[1])

    if img2statistics.__contains__(geom):

        img2statistics[geom]['{}-{}'.format(rougness, specular)] =  float(label)

    else:

        img2statistics[geom] = {'{}-{}'.format(rougness, specular) : float(label)}
    

is_labels.close()

json_data = []

for f in tqdm(range(len(files))):

    f_line = files[f]
    ill, geom, mat = f_line.split('][')
    ill = ill[1:]
    mat = mat.split(']')
    specular = float(mat[0].split('-')[2])
    rougness = float(mat[0].split('-')[1])
    mat = mat[0].split('-')[3]

    ill = str(ill_dict[ill]) + '-' + ill[:-3]

    d = {}
    d["Specular"] = specular
    d["Rougness"] = rougness
    d["Linear"] = get_label_formula(rougness,specular)
    d["Satistics"] = img2statistics[geom]['{}-{}'.format(rougness, specular)] 
    d["Industry"] = get_label_industry(rougness, specular, img2color[f_line])
    d["name"] = f_line
    d["material"] = mat
    d["illumination"] = ill
    d["geometry"] = geom

    json_data.append(d)

with open('data_continuous.json', 'r') as f:

    json.dump(json_data, f)
    output = open('weak_labels.txt', 'w')
    output.write('name, model, statistics, industry\n')
    dataset = json.load(f)
    for data in dataset:

        output.write(str(data['name'])+', ')
        output.write(str(data['Linear'])+', ')
        output.write(str(data['Satistics'])+', ')
        output.write(str(data['Industry'])+'\n')

    output.close()