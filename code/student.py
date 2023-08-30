import cv2
import numpy as np
import dlib
from scipy.spatial import Delaunay

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(detector, predictor, img):
    
    '''
    This function first use `detector` to localize face bbox and then use `predictor` to detect landmarks (68 points, dtype: np.array).
    
    Inputs: 
        detector: a dlib face detector
        predictor: a dlib landmark detector, require the input as face detected by detector
        img: input image
        
    Outputs:
        landmarks: 68 detected landmark points, dtype: np.array

    '''
    
    rects = detector(img, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
 
    landmarks=np.array([[p.x, p.y] for p in predictor(img, rects[0]).parts()])
    
    return landmarks

def get_face_mask(img, landmarks):
    
    '''
    This function gets the face mask according to landmarks.
    
    Inputs: 
        img: input image
        landmarks: 68 detected landmark points, dtype: np.array
        
    Outputs:
        convexhull: face convexhull
        mask: face mask 

    '''
    
    #TODO: Implement this function!  

    # 创建一个与输入图像大小相同的空白遮罩图像
    mask = np.zeros_like(img,dtype='uint8')

    # 获取面部凸包
    convexhull = cv2.convexHull(landmarks)

    # 在遮罩图像上绘制填充的凸包多边形
    mask = cv2.fillConvexPoly(mask, convexhull, (255, 255, 255))
    
    return convexhull, mask

def get_delaunay_triangulation(landmarks, convexhull):
    
    '''
    This function gets the face mesh triangulation according to landmarks.
    
    Inputs: 
        landmarks: 68 detected landmark points, dtype: np.array
        convexhull: face convexhull
        
    Outputs:
        triangles: face triangles 
    '''
    
    #TODO: Implement this function!
    
    # 将凸包的点坐标转换为整数类型
    convexhull = np.array(convexhull, dtype=np.int32)

    # 将凸包内的点索引提取出来
    convex_indices = [i for i, point in enumerate(landmarks) if tuple(point) in convexhull]

    # 获取凸包内的点坐标
    convex_points = landmarks[convex_indices]

    # 将凸包内的点与所有地标点合并
    points = np.vstack((landmarks, convex_points))

    # 使用Delaunay三角剖分算法计算三角形索引
    tri = Delaunay(convex_points)

    # 获取三角形索引
    triangles = convex_points[tri.simplices]
    
    return triangles

def transformation_from_landmarks(target_landmarks, source_landmarks):
    '''
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    
    Inputs: 
        target_landmarks: 68 detected landmark points of the target face, dtype: np.array
        source_landmarks: 68 detected landmark points of the source face that need to be warped, dtype: np.array
        
    Outputs:
        triangles: face triangles 
    '''
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    
    #TODO: Implement this function!
    
    points1 = target_landmarks.astype(np.float64)
    points2 = source_landmarks.astype(np.float64)
 
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
 
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
 
    U, S, Vt = np.linalg.svd(np.dot(points1.T , points2))
 
    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (np.dot(U , Vt)).T
 
    M=np.vstack([np.hstack([s2 / s1 * R, (c2.T - np.dot(s2 / s1 * R, c1.T))[:, np.newaxis]]), np.array([[0., 0., 1.]])])
    
    return M

def warp_img(img, M, target_shape):
    '''
    This function utilizes the affine transformation matrix M to transform the img.
    
    Inputs: 
        img: input image (np.array) need to be warped.
        M: affine transformation matrix.
        target_shape: the image shape of target image
        
    Outputs:
        warped_img: warped image.
    
    '''
    
    #TODO: Implement this function!
    output_img = np.zeros(target_shape, dtype=img.dtype)
    cv2.warpAffine(img, M[:2], (target_shape[1], target_shape[0]), dst=output_img, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)

    return output_img