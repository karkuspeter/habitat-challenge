import cv2
import json
import numpy as np


def get_all_floors_from_file(scene_id):
    floor_height_path = '../GibsonSim2RealChallenge/gibson-challenge-data/dataset/%s/floors.txt'%scene_id
    with open(floor_height_path, 'r') as f:
        floors = sorted(list(map(float, f.readlines())))
    print(scene_id + ' floors', floors)
    return floors


def get_floor(height, floor_heights):
    floor = np.argmin(np.abs(np.array(floor_heights) - height))
    return floor


def get_floor_from_file(sceen_id, height):
    floor_heights = get_all_floors_from_file(sceen_id)
    return get_floor(height, floor_heights)


def load_map_from_file(scene_id, height=None, floor=None, map_name="map"):
    base_path = './data/habitat/maps/'

    if floor is None:
        assert height is not None
        floor_filename = base_path + '%s_floors.json' % scene_id
        with open(floor_filename, 'r') as file:
            floor_heights = json.load(file)['floor_heights']
        floor = get_floor(height, floor_heights)

    map_filename = base_path + scene_id + '_%d_%s.png' % (floor, map_name)
    print (map_filename)

    global_map = cv2.imread(map_filename, cv2.IMREAD_UNCHANGED)
    if global_map is None:
        raise ValueError('Image could not be loaded from %s' % map_filename)
    global_map = np.atleast_3d(global_map)

    return global_map