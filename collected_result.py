import numpy as np
from os import path
from typing import List
from radiosity_data import DataSlice
from tqdm import tqdm
from openexr_numpy import imwrite


class CollectedResult:

    def __init__(self, collected_data_dir):
        self.collected_data_dir = collected_data_dir
        with open(path.join(collected_data_dir, 'camera.csv')) as camera_file:
            # 0001, camera_x, camera_y, camera_z, ...
            self.slices: List[DataSlice] = []
            camera_lines = camera_file.readlines()
            self.camera_data = []
        for line in tqdm(camera_lines, desc="Loading data slices"):
            camera_data_line = [s.strip() for s in line.strip().split(',')]
            slice_id = camera_data_line[0]
            camera_pos = np.array([float(camera_data_line[1]), float(camera_data_line[2]), float(camera_data_line[3])])
            self.slices.append(DataSlice(
                camera_pos,
                path.join(collected_data_dir, slice_id, 'posw.exr'),
                path.join(collected_data_dir, slice_id, 'color.exr')
            ))
            self.camera_data.append((slice_id, camera_pos))


    def prepare_training_data(self):
        return_data = {
            'posw': [],
            'direction': [],
            'color': []
        }

        for slice in tqdm(self.slices, desc="Collapsing data slices"):
            return_data['posw'].append(slice.posw_collapsed)
            return_data['direction'].append(slice.direction_collapsed)
            return_data['color'].append(slice.color_collapsed)

        return {
            'posw': np.concatenate(return_data['posw']),
            'direction': np.concatenate(return_data['direction']),
            'color': np.concatenate(return_data['color'])
        }


    def write_perdict_images(self, perdict_func):
        for camera_info, slice in tqdm(zip(self.camera_data, self.slices)):
            rendered_image = perdict_func(slice)
            output_path = path.join(self.collected_data_dir, camera_info[0], 'predict.exr')
            imwrite(output_path, rendered_image)


if __name__ == "__main__":
    collected_result = CollectedResult('../RadiosityCollecterOutput')
    collected_result.slices[0].print_dimensions()
