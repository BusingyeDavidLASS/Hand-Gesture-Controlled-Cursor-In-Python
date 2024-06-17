import numpy as np

def get_angle(a, b, c):
    radians = np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(a.y - b.y, a.x - b.x)
    angle = np.abs(np.degrees(radians))
    return angle

def get_distance(landmark_list):
    if len(landmark_list) < 2:
        return None  # Handle case where there are fewer than 2 landmarks

    # Extract x and y coordinates of the first two landmarks
    x1, y1 = landmark_list[0].x, landmark_list[0].y
    x2, y2 = landmark_list[1].x, landmark_list[1].y

    # Calculate Euclidean distance between the two points
    L = np.hypot(x2 - x1, y2 - y1)

    # Interpolate distance to fit desired range (0, 1000)
    return np.interp(L, [0, 1], [0, 1000])
