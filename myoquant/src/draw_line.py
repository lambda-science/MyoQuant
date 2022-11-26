import math


def line_equation(x1, y1, x2, y2):
    # Poor man's handling of the case where the equation is infinite
    if x2 - x1 == 0 and y2 > y1:
        m = 9999
    elif x2 - x1 == 0 and y1 > y2:
        m = -9999
    else:
        m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    if math.isinf(m) or math.isinf(b):
        raise ValueError("Line equation is infinite")
    return m, b


def calculate_intersection(m, b, image_dim=(256, 256)):
    x = [0, image_dim[1]]
    y = [0, image_dim[0]]
    results = []
    for i in x:
        intersect = m * i + b
        if intersect >= 0 and intersect < image_dim[0]:
            results.append((i, intersect))
    for i in y:
        intersect = (i - b) / m
        if intersect >= 0 and intersect < image_dim[1]:
            results.append((intersect, i))
    if results == []:
        raise ValueError("No intersection found")
    return results


def calculate_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def calculate_closest_point(x1, y1, intersections):
    distances = []
    for coords in intersections:
        distances.append(calculate_distance(x1, y1, coords[0], coords[1]))
    return intersections[distances.index(min(distances))]
