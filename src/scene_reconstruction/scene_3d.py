"""Module for processing 2D image data for scene reconstruction."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scene_3d_utils import draw_lines


class SceneReconstruction3D:
    """The Class contains public and private fuction used for extracting key points from images, estimating optical flow
     and plotting 3D points."""
    def __init__(self, K, dist):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.d = dist
        self.img_1 = None
        self.img_2 = None

    def load_image_pair(self, image_path_1, image_path_2, down_scale=True):
        """
        Load stereo images, converting them to grayscale and correcting distortion error.
        :param image_path_1: Path to image one
        :param image_path_2: Path to image two
        :param down_scale: Option to rescale image
        :return: None
        """
        self.img_1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
        self.img_2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)

        if len(self.img_1.shape) == 2:
            self.img_1 = cv2.cvtColor(self.img_1, cv2.COLOR_GRAY2BGR)
            self.img_2 = cv2.cvtColor(self.img_2, cv2.COLOR_GRAY2BGR)

        target_width = 600
        if down_scale and self.img_1.shape[1] > target_width:
            while self.img_1.shape[1] > 2 * target_width:
                self.img_1 = cv2.pyrDown(self.img_1)
                self.img_2 = cv2.pyrDown(self.img_2)

        self.img_1 = cv2.undistort(self.img_1, self.K, self.d)
        self.img_2 = cv2.undistort(self.img_2, self.K, self.d)

    def _extract_key_points_flow(self):
        """
        Extract key points from images using FAST feature detector.
        :return: None
        """
        fast = cv2.FastFeatureDetector_create()
        first_key_points = fast.detect(self.img_1, None)

        first_key_list = [i.pt for i in first_key_points]
        first_key_arr = np.array(first_key_list).astype(np.float32)

        second_key_arr, status, err = cv2.calcOpticalFlowPyrLK(self.img_1, self.img_2, first_key_arr, None)
        condition = (status == 1) * (err < 4.)

        concat = np.concatenate((condition, condition), axis=1)
        first_match_points = first_key_arr[concat].reshape(-1, 2)
        second_match_points = second_key_arr[concat].reshape(-1, 2)

        self.match_pts_1 = first_match_points
        self.match_pts_2 = second_match_points

    def plot_optic_flow(self):
        """
        Plot optical flow from points extracted from stereo images.
        :return: None
        """
        img = self.img_1
        self._extract_key_points_flow()

        for i in range(len(self.match_pts_1)):
            a, b = self.match_pts_1[i]
            c, d = self.match_pts_2[i]
            cv2.line(img, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 2)
            cv2.circle(img, (int(a), int(b)), 5, (0, 255, 0), -1)

        cv2.imshow("Flow", img)
        cv2.waitKey()

    def plot_rectified_stereo_images(self):
        """
        Plot rectified stereo images.
        :return:
        """
        self._extract_key_points_flow()

        pts1 = np.int32(self.match_pts_1)
        pts2 = np.int32(self.match_pts_2)

        fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # We select only inlier points
        pts1 = pts1[inliers.ravel() == 1]
        pts2 = pts2[inliers.ravel() == 1]

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(
            pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = draw_lines(self.img_1, self.img_2, lines1, pts1, pts2)

        # Find epilines corresponding to points in the left image (first image) and
        # drawing its lines on the right image
        lines2 = cv2.computeCorrespondEpilines(
            pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = draw_lines(self.img_1, self.img_2, lines2, pts2, pts1)

        plt.subplot(121), plt.imshow(img5)
        plt.subplot(122), plt.imshow(img3)
        plt.suptitle("Epilines in both images")
        plt.show()

        h1, w1, _ = self.img_1.shape
        h2, w2, _ = self.img_2.shape
        _, H1, H2 = cv2.stereoRectifyUncalibrated(
            np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
        )
        img1_rectified = cv2.warpPerspective(self.img_1, H1, (w1, h1))
        img2_rectified = cv2.warpPerspective(self.img_2, H2, (w2, h2))
        cv2.imwrite("rectified_1.png", img1_rectified)
        cv2.imwrite("rectified_2.png", img2_rectified)

        # Draw the rectified images
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes[0].imshow(img1_rectified, cmap="gray")
        axes[1].imshow(img2_rectified, cmap="gray")
        axes[0].axhline(250)
        axes[1].axhline(250)
        axes[0].axhline(450)
        axes[1].axhline(450)
        plt.suptitle("Rectified images")
        plt.savefig("rectified_images.png")
        plt.show()

    def draw_epipolar_lines(self):
        """
        Draw epiplor from stereo frames
        :return:
        """
        sift = cv2.SIFT_create()

        # detect and compute the key points and descriptors with ORB
        kp1, des1 = sift.detectAndCompute(self.img_1, None)
        kp2, des2 = sift.detectAndCompute(self.img_2, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        matches_mask = [[0, 0] for i in range(len(matches))]

        good = []
        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                # keep this keypoint pair
                matches_mask[i] = [1, 0]
                good.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)

        # Draws the keypoint matches between both pictures
        # Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matches_mask[300:500],
                           flags=cv2.DrawMatchesFlags_DEFAULT)

        keypoint_matches = cv2.drawMatchesKnn(
            self.img_1, kp1, self.img_2, kp2, matches[300:500], None, **draw_params)
        cv2.imshow("Keypoint matches", keypoint_matches)
        cv2.waitKey(0)
