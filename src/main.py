"""Main sfm module, usage: $python3 ./src/main.py"""
import numpy as np
from scene_reconstruction.scene_3d import SceneReconstruction3D

# Add argparser for folder to image pair


def main():
    K = np.array([[2759.48/4, 0, 1520.69/4, 0, 2764.16/4, 1006.81/4, 0, 0, 1]]).reshape(3, 3)
    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)

    scene = SceneReconstruction3D(K, d)
    scene.load_image_pair("./data/P1060102.JPG", "./data/P1060103.JPG")
    scene.plot_optic_flow()
    scene.draw_epipolar_lines()
    scene.plot_rectified_stereo_images()

    # scene.plot_point_cloud()
    # Todo


if __name__ == "__main__":
    main()

