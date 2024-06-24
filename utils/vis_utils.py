import cv2

def vis_bbox(img, bbox):
    img = img.copy()
    color, thickness = (0, 255, 0), 1

    if len(bbox) == 4:
        x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        pos1 = (x_min, y_min)
        pos2 = (x_min, y_max)
        pos3 = (x_max, y_min)
        pos4 = (x_max, y_max)

        img = cv2.line(img, pos1, pos2, color, thickness) 
        img = cv2.line(img, pos1, pos3, color, thickness) 
        img = cv2.line(img, pos2, pos4, color, thickness) 
        img = cv2.line(img, pos3, pos4, color, thickness) 

    return img