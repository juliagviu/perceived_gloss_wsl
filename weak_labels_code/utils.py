import math
from utils import *

ANLE = 20
RADIO = 1
IRRADIANCE= 1.0
REFLECTANCE_GLASS = 0.005
IOR = 1.567
G_MAX = 275.9596652273564

def get_label_formula(rougness,specular):

    # To get values as floats
    # return  4 * (-(rougness * rougness) + 0.5) * 1.2 +  specular* 0.95
    return math.floor( 4 * (-(rougness * rougness) + 0.5) * 1.2 +  specular* 0.95)

def get_label_industry(rougness, specular, reflectance_disney):

    light_position = sphere2car (RADIO, 180, ANLE)
    sample_position = sphere2car (RADIO, 359, ANLE)
    diff =  np.array([0,0,0]) - sample_position
    scene_disney = mi.load_dict(create_scene_disney(light_position, IRRADIANCE, reflectance_disney, rougness, specular)) 
    scene_glass = mi.load_dict(create_scene_rough_glass(light_position, IRRADIANCE, REFLECTANCE_GLASS, IOR))
    radiance_meter = load_sensor(sample_position, diff)
    measure_disney = np.array(mi.render(scene_disney, sensor=radiance_meter) )[0,0,0] / np.cos(np.radians(ANLE))
    measure_glass = np.array(mi.render(scene_glass, sensor=radiance_meter) )[0,0,0] / np.cos(np.radians(ANLE))
    measure_glass_log = np.log10(np.array(measure_glass) + 1.0)
    measure_disney_log = np.log10(np.array(measure_disney) + 1.0)
    G_log = 100 * (measure_disney_log / measure_glass_log)

    # To get values as floats
    # return  ((G_log / G_MAX) * 6) + 1
    return  math.ceil((G_log / G_MAX) * 7)

import numpy as np
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
from mitsuba import ScalarTransform4f as T

def sphere2car (r,phi, theta):
    phi = np.radians(phi)
    theta = np.radians(theta)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    x = r * np.sin(theta) * np.cos(phi)
    return np.array([y,x,z])

def load_sensor(point,dir):
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.

    return mi.load_dict({
        'type': 'radiancemeter',
        'origin' : point,
        'direction' : dir,
        'film': {
            'type': 'hdrfilm',
            'width': 1,
            'height': 1,
            'rfilter': {
                'type': 'box',
            },
            'pixel_format': 'luminance',
            'component_format' : 'float32'
        },
    })

def load_camera(point):

    return mi.load_dict({
        'type': 'perspective',
        'fov': 40,
        'to_world': T.look_at(
            origin=point.tolist(),
            target=[0, 0, 0],
            up = [0, 0, 1]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 256
        },
        'film': {
            'type': 'hdrfilm',
            'width': 256,
            'height': 256,
            'rfilter': {
                'type': 'tent',
            },
            'pixel_format': 'rgb',
        },
    })

def create_scene_diffuse(point, irradiance, reflectance):

    target = np.array([0,0,0])
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path'
        },
        'directional_emitter': {
            'type': 'directional',
            'irradiance': irradiance,
            'direction' : target - point
        },

        'plane_1': {
            'type': 'rectangle',
                'material': {
                    'type': 'diffuse',
                    'reflectance' : reflectance
            }       
        }
    }

    return scene_dict

def create_scene_disney(point, irradiance, reflectance,roughness_value, specular_value):

    target = np.array([0,0,0])
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path'
        },
        'directional_emitter': {
            'type': 'directional',
            'irradiance': irradiance,
            'direction' : target - point
        },

        'plane_1': {
            'type': 'rectangle',
                'material': {
                    'type': 'principled',
                    'base_color' :  {
                        'type': 'rgb',
                        #'value': [.82, .67, .16]
                        'value': reflectance
                    },
                    'specular': specular_value,
                    'roughness': roughness_value,
            }       
        }
    }

    return scene_dict

def create_scene_black_glass(point, irradiance, reflectance, IOR):

    target = np.array([0,0,0])
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path'
        },
        'directional_emitter': {
            'type': 'directional',
            'irradiance': irradiance,
            'direction' : target - point
        },

        'plane_1': {
            'type': 'rectangle',
                'material': {
                    'type': 'plastic',
                    'diffuse_reflectance' :  {
                        'type': 'rgb',
                        'value': reflectance
                    },
                    'specular_reflectance' :  {
                        'type': 'rgb',
                        'value': 1.0
                    },
                    'int_ior': IOR
            }       
        }
    }

    return scene_dict

def create_scene_rough_glass(point, irradiance, reflectance, IOR):

    target = np.array([0,0,0])
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path'
        },
        'directional_emitter': {
            'type': 'directional',
            'irradiance': irradiance,
            'direction' : target - point
        },

        'plane_1': {
            'type': 'rectangle',
                'material': {
                    'type': 'roughplastic',
                    'diffuse_reflectance' :  {
                        'type': 'rgb',
                        'value': reflectance
                    },
                    'specular_reflectance' :  {
                        'type': 'rgb',
                        'value': 1.0
                    },
                    'int_ior': IOR,
                    'alpha' : 0.025,
                    'distribution': 'ggx'
            }       
        }
    }

    return scene_dict

reflectance_disney = [
[0.1071192024, 0.0725399385, 0.0860190437],
[0.1914998957, 0.3051689153, 0.2397755217],
[0.4877096386, 0.4254469366, 0.4555930847],
[0.9779333461, 0.93717365, 0.9518877935],
[0.761286398, 0.5304022819, 0.7669616965],
[0.0960104304, 0.6963393501, 0.953527772],
[0.1298824869, 0.1726687592, 0.1806144797],
[0.3141176639, 0.2887460073, 0.2530820744],
[0.9975776808, 0.9711086707, 0.9786270464],
[0.4993750788, 0.4248797921, 0.4214805793],
[0.7090516879, 0.7278724555, 0.9119288247],
[0.441076398, 0.056560347, 0.1667753756],
[0.468402033, 0.7142032759, 0.9308705117],
[0.1357468763, 0.3094995354, 0.3643081291],
[0.0704022525, 0.0418774546, 0.0064142459],
[0.0444825744, 0.2143903684, 0.102333937],
[0.0155677478, 0.0521311161, 0.0337358497],
[0.1046812719, 0.0430337099, 0.0927879701],
[0.1068375808, 0.3340246281, 0.3886751259],
[0.5113411518, 0.791857824, 0.8863819827],
[0.2157307425, 0.1644206797, 0.1263844582],
[0.1563335842, 0.1693410586, 0.1066070704],
[0.9075719799, 0.8875989685, 0.8904036028],
[0.0863961825, 0.0461723219, 0.0650701949],
[0.0265142007, 0.1617935301, 0.1475381677],
[0.4766809424, 0.1557260832, 0.2907278195],
[0.8506150164, 0.9980585615, 0.8918611927],
[0.9419227251, 0.9057413521, 0.8913111289],
[0.7245448974, 0.8169126748, 0.8094700896],
[0.4543101706, 0.4532654025, 0.4501017233],
[0.2994076947, 0.5972409282, 0.5988994733],
[0.0090863786, 0.0010384555, 0.0059133959],
[0.1802606218, 0.1011542954, 0.0440853007],
[0.1850418749, 0.6316361914, 0.6511941103],
[0.0887956009, 0.0734085861, 0.0929088303],
[0.0063698276, 0.0256168161, 0.0272865166],
[0.1575995575, 0.2156213393, 0.2340130445],
[0.3520851983, 0.3657087394, 0.2031605932],
[0.918253854, 0.1318204205, 0.7765490853],
[0.8502753535, 0.9448193141, 0.8928368652],
[0.9392981422, 0.9341823429, 0.9567048339],
[0.0297583489, 0.5889532272, 0.3777894403],
[0.9741056862, 0.9027656022, 0.8878741796],
[0.0845946802, 0.1216190157, 0.0928269345],
[0.9052268839, 0.8933124261, 0.8952055214],
[0.8612400321, 0.9664192084, 0.7489869177],
[0.3120449292, 0.3177884228, 0.2091313268],
[0.5753549492, 0.5826501627, 0.2371830574],
[0.5505245702, 0.5854766334, 0.6444930518],
[0.7882365032, 0.3714379065, 0.5591894462]
]