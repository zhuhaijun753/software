#!/usr/bin/env python3

import math
import time

from collections import namedtuple
from functools import reduce
from itertools import permutations

import cv2
import numpy as np

import shm

from vision.modules.base import ModuleBase
from vision.vision_common import green, red, white, yellow
from vision import options
from color_balance import balance

#XXX Simulator constants
options = [ options.BoolOption('debug', False),
            options.IntOption('hls_h_min', 0, 0, 255),
            options.IntOption('hls_h_max', 120, 0, 255),
            options.IntOption('hls_s_min', 100, 0, 255),
            options.IntOption('hls_s_max', 255, 0, 255),
            options.IntOption('lab_l_min', 0, 0, 255),
            options.IntOption('lab_l_max', 0, 0, 255),
            options.IntOption('lab_a_min', 30, 0, 255),
            options.IntOption('lab_a_max', 140, 0, 255),
            options.IntOption('lab_b_min', 125, 0, 255),
            options.IntOption('lab_b_max', 230, 0, 255),
            options.IntOption('erode_kernel_size', 9, 1, 151),
            options.IntOption('dilate_kernel_size', 9, 1, 151),
            options.IntOption('max_error', 15000, 100, 100000),
            options.DoubleOption('min_circularity', 0.65, 0.1, 0.95),
            options.DoubleOption('min_percent_frame', 0.0002, 0, 0.01) ]

# L, A, B; modify A while tuning, keep L and B constant
RED_REF = (100, 150, 170)
GREEN_REF = (100, 80, 170)
YELLOW_REF = (100, 115, 170)

#XXX Teagle constants
#options = [ options.BoolOption('debug', False),
#            options.IntOption('hls_h_min', 0, 0, 255),
#            options.IntOption('hls_h_max', 120, 0, 255),
#            options.IntOption('hls_s_min', 0, 0, 255),
#            options.IntOption('hls_s_max', 200, 0, 255),
#            options.IntOption('lab_l_min', 0, 0, 255),
#            options.IntOption('lab_l_max', 0, 0, 255),
#            options.IntOption('lab_a_min', 30, 0, 255),
#            options.IntOption('lab_a_max', 140, 0, 255),
#            options.IntOption('lab_b_min', 128, 0, 255),
#            options.IntOption('lab_b_max', 230, 0, 255),
#            options.IntOption('erode_kernel_size', 9, 1, 151),
#            options.IntOption('dilate_kernel_size', 9, 1, 151),
#            options.IntOption('max_error', 15000, 100, 100000),
#            options.DoubleOption('min_circularity', 0.65, 0.1, 0.95),
#            options.DoubleOption('min_percent_frame', 0.0002, 0, 0.01) ]
#
#RED_REF = (90, 130, 170)
#GREEN_REF = (90, 80, 170)
#YELLOW_REF = (90, 100, 170)

CONTOUR_CIRCULARITY_HEURISTIC_LIMIT = 50
CONTOUR_SCALED_HEURISTIC_LIMIT = 3
CONTOUR_CLASSIFICATION_LIMIT = 3
ContourAreaData = namedtuple('ContourAreaData', ['contour', 'area'])
ContourScoreData = namedtuple('ContourScoreData', ['contour', 'area', 'circularity', 'score', 'center', 'radius'])

INVALID_BUOY_ERROR = 1e99

class BuoyData:
    def __init__(self, shm_group, color, reference_color):
        self.shm_group = shm_group
        self.results = shm_group.get()
        self.color = color
        self.reference_color = reference_color
        self.reset_frame_state()

    def reset_frame_state(self):
        self.contour = None

    def zero_results(self):
        self.results.heuristic_score = 0
        self.results.probability = 0
        self.results.area = 0
        self.results.percent_frame = 0

    def set_results(self, image_size, contour_list):
        assert self.contour is not None # This must be called after contour has been set.
        self.results.center_x = int(self.contour.center[0])
        self.results.center_y = int(self.contour.center[1])
        self.results.top_x = int(self.contour.center[0])
        self.results.top_y = int(self.contour.center[1] - self.contour.radius / 2)
        self.results.area = self.contour.area
        self.results.heuristic_score = self.contour.score
        self.results.percent_frame = 100 * self.contour.area / image_size
        self.results.probability = self.contour.score / reduce(lambda acc, x: acc + x.score, contour_list, 0)
        # TODO XXX HAX Please fix this probabilities.
        self.results.probability = max(self.results.probability, 0.5)

    def set_shm_group(self):
        self.shm_group.set(self.results)

RED_BUOY = BuoyData(shm.red_buoy_results, red, RED_REF)
GREEN_BUOY = BuoyData(shm.green_buoy_results, green, GREEN_REF)
YELLOW_BUOY = BuoyData(shm.yellow_buoy_results, yellow, YELLOW_REF)
BUOYS = [RED_BUOY, GREEN_BUOY, YELLOW_BUOY]

class Buoys(ModuleBase):
    def calculate_error(self, buoy, candidate):
        error_vec = np.array(buoy.reference_color) - \
                    np.array(candidate[1])
        buoy_error = error_vec.dot(error_vec)
        if buoy_error > self.options['max_error']:
            return INVALID_BUOY_ERROR
        else:
            return buoy_error

    def classify_buoys(self, buoys, candidates):
        """
            buoys is a list of BuoyData objects
            candidates is a list of length at most 3 of (candidate, color) tuples
        """
        # Possible mappings for the # of candidates to the 3 buoys
        possible_mappings = permutations(buoys, len(candidates))
        assignments = []
        max_buoys_classified = 0
        for mapping in possible_mappings:
            total_error = 0
            classification = {buoy: None for buoy in mapping}
            buoys_classified = 0
            # Iterate through one-to-one mappings of buoys to candidates
            for i in range(len(candidates)):
                error = self.calculate_error(mapping[i], candidates[i])
                if error >= INVALID_BUOY_ERROR:
                    classification[mapping[i]] = None
                else:
                    classification[mapping[i]] = candidates[i][0]
                    total_error += error
                    buoys_classified += 1
            #print("Considering: " + str([buoy.color for buoy in classification]))
            #print(total_error)
            if buoys_classified >= max_buoys_classified:
                max_buoys_classified = buoys_classified
                assignments.append((total_error, buoys_classified, classification))
        # Consider assignments with the most number of classified buoys
        assignments = filter(lambda x: x[1] == max_buoys_classified, assignments)
        min_error = self.options['max_error']
        optimal_assignment = (0, 0, {buoy: None for buoy in buoys})
        for assignment in assignments:
            if assignment[0] < min_error:
                min_error = assignment[0]
                optimal_assignment = assignment
        print(len(optimal_assignment[2]), min_error)
        #print([buoy.color for buoy in [k for (k,v) in optimal_assignment[2].items() if
        #    v is not None]])
        return optimal_assignment

    def process(self, mat):
        start_time = time.time()
        self.post('orig', mat)
        #mat = balance(mat)
        #self.post('balanced', mat)
        hls = cv2.cvtColor(mat, cv2.COLOR_BGR2HLS)
        hls_split = cv2.split(hls)
        lab = cv2.cvtColor(mat, cv2.COLOR_BGR2LAB)
        lab_split = cv2.split(lab)

        #a_threshed = cv2.inRange(lab_split[1], self.options['lab_a_min'],
        #                                       self.options['lab_a_max'])
        #self.post('a_threshed', a_threshed)

        b_threshed = cv2.inRange(lab_split[2], self.options['lab_b_min'],
                                               self.options['lab_b_max'])
        if self.options['debug']:
            self.post('b_threshed', b_threshed)

        s_threshed = cv2.inRange(hls_split[2], self.options['hls_s_min'],
                                               self.options['hls_s_max'])
        if self.options['debug']:
            self.post('s_threshed', s_threshed)

        h_threshed = cv2.inRange(hls_split[0], self.options['hls_h_min'],
                                               self.options['hls_h_max'])
        if self.options['debug']:
            self.post('h_threshed', h_threshed)

        threshed = s_threshed & b_threshed & h_threshed

        morphed = cv2.erode(threshed,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                        (self.options['erode_kernel_size'],
                         self.options['erode_kernel_size'])))
        morphed = cv2.dilate(morphed,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                        (self.options['dilate_kernel_size'],
                         self.options['dilate_kernel_size'])))

        if self.options['debug']:
            self.post('threshed', threshed)
            self.post('morphed', morphed)

        _, contours, hierarchy = cv2.findContours(morphed.copy(),
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)

        image_size = mat.shape[0] * mat.shape[1]

        contourAreas = []
        for contour in contours:
            contourArea = cv2.contourArea(contour)
            if contourArea >= image_size * self.options['min_percent_frame']:
                contourAreas.append(ContourAreaData(contour, contourArea))
        # Sort contours in descending order by their areas
        contourAreas = sorted(contourAreas, key=lambda x: -x.area)[:CONTOUR_CIRCULARITY_HEURISTIC_LIMIT]

        contourScores = []
        for contourArea in contourAreas:
            center, radius = cv2.minEnclosingCircle(contourArea.contour)
            circularity = contourArea.area / (math.pi * radius ** 2)
            if circularity >= self.options['min_circularity']:
                # Calculate a heuristic score defined by
                # (contour cirularity)^2 * (contour area)
                heuristic_score = (circularity**2) * contourArea.area
                contourScores.append(ContourScoreData(contourArea.contour,
                                     contourArea.area, circularity, heuristic_score, center, radius))
        # Sort contours in descending order by their heuristic score
        contourScores = sorted(contourScores, key=lambda x: -x.score)[:CONTOUR_SCALED_HEURISTIC_LIMIT]

        if contourScores:
            # Reduce score of top contour in image to lower ranking of buoy
            # internal reflection in the surface of the water
            #topContourIdx = 0
            #topContourY = mat.shape[1]
            #for idx in range(len(contourScores)):
            #    if contourScores[idx].center[1] < topContourY: # Zero is top-left of image
            #        topContourY = contourScores[idx].center[1]
            #        topContourIdx = idx
            #topContour = contourScores[topContourIdx]
            #contourScores[topContourIdx] = topContour._replace(score=topContour.score / 2)

            if self.options['debug']:
                contoursMat = mat.copy()
                cv2.drawContours(contoursMat, [cand.contour for cand in contourScores], -1, white, 3)

            buoyContours = sorted(contourScores, key=lambda x: -x.score)[:CONTOUR_CLASSIFICATION_LIMIT]
            total_mask = np.zeros(mat.shape, np.uint8)
            buoy_results = []
            # Only look at top contour candidates.
            for i, buoy_candidate in enumerate(buoyContours):
                mask = np.zeros(lab.shape, np.uint8)
                cv2.drawContours(mask, [buoy_candidate.contour], 0, white, cv2.FILLED)
                if i == 0:
                    cv2.drawContours(total_mask, [buoy_candidate.contour], 0, green, 1)
                else:
                    cv2.drawContours(total_mask, [buoy_candidate.contour], 0, white, 1)

                just_buoy = cv2.bitwise_and(lab, mask)
                just_buoy_split = cv2.split(just_buoy)

                l_ch = just_buoy_split[0]
                a_ch = just_buoy_split[1]
                b_ch = just_buoy_split[2]

                l_buoy_area = max(np.count_nonzero(l_ch), 1)
                a_buoy_area = max(np.count_nonzero(a_ch), 1)
                b_buoy_area = max(np.count_nonzero(b_ch), 1)
                avg_l = np.sum(l_ch) / l_buoy_area
                avg_a = np.sum(a_ch) / a_buoy_area
                avg_b = np.sum(b_ch) / b_buoy_area
                print(avg_l, avg_a, avg_b)
                buoy_results.append((buoy_candidate, (avg_l, avg_a, avg_b)))

            # Best fit classification
            [buoy.reset_frame_state() for buoy in BUOYS]
            total_error, buoys_classified, classification = self.classify_buoys(BUOYS, buoy_results[:3])
            #print(buoys_classified)
            for (buoy, contour) in classification.items():
                if contour is not None:
                    buoy.contour = contour
                    buoy.set_results(image_size, contourScores)
                else:
                    buoy.reset_frame_state()
                    buoy.zero_results()

            buoy_contoursMat = mat.copy()
            for buoy in BUOYS:
                if buoy.contour is not None:
                    cv2.drawContours(buoy_contoursMat, [buoy.contour.contour], -1, buoy.color, 6)

            self.post("total_mask", total_mask)
            self.post("All buoys", buoy_contoursMat)

        else:
            [buoy.zero_results() for buoy in BUOYS]

        if self.options['debug'] and contourScores:
            self.post('contours', contoursMat)

        for buoy in BUOYS:
            self.fill_single_camera_direction(buoy.results)
            buoy.set_shm_group()
        end_time = time.time()
        print("Elapsed time: " + str(end_time - start_time))

if __name__ == '__main__':
    Buoys('forward', options)()
