# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np


def fit_triangle_to_mask(tri_1_points, tri_2_points):

    tri_1_points, _ = find_triangle_from_edges(tri_1_points)
    tri_2_points, _ = find_triangle_from_edges(tri_2_points)

    if tri_1_points is None or tri_2_points is None:
        return (None, None, None, None, None, None)

    most_left_point = 100000
    tri_left = None
    tri_right = None
    for point in tri_1_points + tri_2_points:
        if point[0] < most_left_point:
            most_left_point = point[0]
            tri_left = tri_1_points if point in tri_1_points else tri_2_points
            tri_right = tri_2_points if point in tri_1_points else tri_1_points

    # check that tri_left is the left triangle
    if tri_left[0][0] > tri_right[0][0]:
        tri_left, tri_right = tri_right, tri_left

    fem_l, pel_l_o, pel_l_i = define_points(tri_left)
    fem_r, pel_r_o, pel_r_i = define_points(tri_right)

    return fem_l, pel_l_o, pel_l_i, fem_r, pel_r_o, pel_r_i


def find_triangle_from_edges(points):
    # Means we are already passing in a processed triangle
    if len(points) == 3:
        triangle = np.array(points)
        return triangle, cv2.contourArea(triangle)

    contours = np.array([points], dtype=np.int32)

    if len(contours) == 0:
        return None, 0  # No contours found

    # Approximate the largest contour to a polygon
    epsilon = 0.1 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # Check if the approximated contour has 3 vertices (triangle)
    if len(approx) == 3:
        triangle = np.array([approx[0][0], approx[1][0], approx[2][0]])
        return triangle, cv2.contourArea(triangle)
    else:
        return None, 0  # No triangle found


def define_points(triangle):
    # Convert all points to tuples
    triangle = [(int(point[0]), int(point[1])) for point in triangle]

    # Find the lowest point in the triangle
    lowest_point = max(triangle, key=lambda point: point[1])
    triangle.remove(lowest_point)

    # Find the leftmost point in the triangle
    highest_point = min(triangle, key=lambda point: point[1])
    triangle.remove(highest_point)

    # The last point is the one not picked
    remaining_point = triangle[0]

    return lowest_point, highest_point, remaining_point
