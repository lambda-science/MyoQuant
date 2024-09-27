"""
Module that contains function used in the nuclei analysis to draw line and calculate coordinates
"""
import math

def line_equation(x1, y1, x2, y2):
    """Calculate the straight line equation given two points"""
    # Use abs function to check for infinite slope or intercept
    if abs(x2 - x1) < 1e-9:
        if y2 > y1:
            m = math.inf
        else:
            m = -math.inf
    else:
        m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    if math.isinf(m) or math.isinf(b):
        raise ValueError("Line equation is infinite")
    return m, b


def calculate_intersection(m, b, image_dim=(256, 256)):
    """Calculate the intersection of a line with the image boundaries"""
    x = [0, image_dim[1]]
    y = [0, image_dim[0]]
    results = []
    for i in x:
        intersect = m * i + b
        # Use min and max functions to check intersection points
        if min(y) <= intersect <= max(y):
            results.append((i, intersect))
    for i in y:
        intersect = (i - b) / m
        if min(x) <= intersect <= max(x):
            results.append((intersect, i))
    if not results:
        raise ValueError("No intersection found")
    return results


def calculate_distance(x1, y1, x2, y2):
    """Calculate the distance between two points"""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def calculate_closest_point(x1, y1, intersections):
    """Calculate the closest intersection point to the first point"""
    # Use key parameter in min function to directly get the tuple with minimum distance
    closest = min(intersections, key=lambda p: calculate_distance(x1, y1, p[0], p[1]))
    return closest
