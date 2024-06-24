import os
import cv2
import numpy as np


from deepface import DeepFace


dataset_name = 'pw3d'
face_det_type = 'all' # ['top1', 'all']


if dataset_name == 'pw3d':
    root_path = 'datasets/PW3D/data/faceFiles'

    sequence_names = os.listdir(root_path)

    for each_seq in sequence_names:
        seq_path = os.path.join(root_path, each_seq)
        seq_crop_img_path = os.path.join(seq_path, 'crop_img')
        if face_det_type == 'top1':
            face_occ_img_path = os.path.join(seq_path, 'face_occ_img')
        elif face_det_type == 'all':
            face_occ_img_path = os.path.join(seq_path, 'face_occ_all_img')
        else:
            import pdb; pdb.set_trace()

        if not os.path.exists(face_occ_img_path):
            os.makedirs(face_occ_img_path)

        image_names = os.listdir(seq_crop_img_path)

        for each_img_name in image_names:
            each_img_path = os.path.join(seq_crop_img_path, each_img_name)
            each_img = cv2.imread(each_img_path)

            #################################### FACE DETECTION ####################################
            face_objs = DeepFace.extract_faces(img_path = each_img_path, detector_backend = 'fastmtcnn', align = True, enforce_detection=False)
            
            if len(face_objs) == 0:
                occ_img = each_img.copy()
                cv2.imwrite(os.path.join(face_occ_img_path, each_img_name), occ_img)
                cv2.imwrite('face_det_debug.png', occ_img)
            elif face_objs[0]['confidence'] == 0:
                occ_img = each_img.copy()
                cv2.imwrite(os.path.join(face_occ_img_path, each_img_name), occ_img)
                cv2.imwrite('face_det_debug.png', occ_img)
            else:
                face_occ_img = np.ones_like(each_img.copy())
                height, width = each_img.shape[0], each_img.shape[1]

                if face_det_type == 'top1':
                    face_objs = face_objs[0:1]

                for each_face_obj in face_objs:
                    face_bbox = each_face_obj['facial_area']

                    # Resize bbox
                    def resize_bbox(face_bbox, resize_ratio):
                        c_x, c_y, w, h = face_bbox['x']+face_bbox['w']//2, face_bbox['y']+face_bbox['h']//2, face_bbox['w'], face_bbox['h']
                        w, h = w*resize_ratio, h*resize_ratio

                        face_bbox['x'], face_bbox['y'] = c_x-w//2, c_y-h//2
                        face_bbox['w'], face_bbox['h'] = w, h
                        return face_bbox
                    
                    face_bbox = resize_bbox(face_bbox, 1.3)

                    # face_vis_img = vis_bbox(orig_crop_img[:,:,::-1], [face_bbox['x'], face_bbox['y'], face_bbox['x']+face_bbox['w'], face_bbox['y']+face_bbox['h']])
                    # cv2.imwrite('face_vis_img.png', face_vis_img)



                    # Occlude face with black patch
                    face_occ_img[max(0, int(face_bbox['y'])): min(height, int(face_bbox['y']+face_bbox['h'])), max(0, int(face_bbox['x'])): min(width, int(face_bbox['x']+face_bbox['w'])), :] = 0.


                occ_img = each_img.copy() * face_occ_img
                cv2.imwrite(os.path.join(face_occ_img_path, each_img_name), occ_img)
                # cv2.imwrite('face_det_debug.png', occ_img)
            #################################### FACE DETECTION ####################################