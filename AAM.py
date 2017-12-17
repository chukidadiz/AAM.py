import numpy as np
import cv2
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# helper
def loadRGBA(path):
    img = cv2.imread(path, -1)
    # if only has rgb, extend it to rgba
    if img.shape[2] == 3:
        rc, gc, bc = cv2.split(img)
        ac = np.ones(bc.shape, dtype=bc.dtype) * 255
        img = cv2.merge((rc, gc, bc, ac))
    return img

def get_pt_filter():
    global mask
    # mask out none jaw face
    mask = np.zeros(68)
    for i in range(3, 14):
        mask[i] = 1
    for i in range(31, 36):
        mask[i] = 1
    for i in range(48, 68):
        mask[i] = 1
    mask = (mask == 1)
    
    return mask

# 回傳一個list [(x1, y1), (x2, y2) ... ]
def get_points(path):
    raw_data = None
    with open(path) as f:
        raw_data = np.array(list(map(lambda x: list(map(float, x.strip('()\n').split(','))), f.readlines()))).flatten()
    
    points = []
    # 留下臉的下半部
    for i in range(len(raw_data)//2):
        if PT_FILTER[i]:
            points.append((int(raw_data[2*i]), int(raw_data[2*i+1])))
    return points

def get_face(path):
    PAD_WIDTH = 5

    points = get_points(path+"_sol")
    
    # 抓靠近目標的區域
    X_MAX = points[0][0]
    X_MIN = points[0][0]
    Y_MAX = points[0][1]
    Y_MIN = points[0][1]
    for p in points:
        X_MAX = max(X_MAX, p[0])
        X_MIN = min(X_MIN, p[0])
        Y_MAX = max(Y_MAX, p[1])
        Y_MIN = min(Y_MIN, p[1])
        
    X_MAX += PAD_WIDTH
    X_MIN -= PAD_WIDTH
    Y_MAX += PAD_WIDTH
    Y_MIN -= PAD_WIDTH
    
    img = loadRGBA(path)
    img = img[Y_MIN:Y_MAX, X_MIN:X_MAX]
    
    for idx, p in enumerate(points):
        points[idx] = (p[0]-X_MIN, p[1]-Y_MIN)
    return points, img

# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def triangulation(path):
    points, img = get_face(path)

    rect = (0, 0, img.shape[1], img.shape[0])
    subdiv  = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)

    tri = []
    for t in subdiv.getTriangleList():
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            pt1idx = points.index(pt1)
            pt2idx = points.index(pt2)
            pt3idx = points.index(pt3)
            tri.append((pt1idx, pt2idx, pt3idx))
    
    return tri

def display_mesh(path, mesh):
    points, img = get_face(path)

    for t in mesh:
        p1, p2, p3 = t
        pt1 = points[p1]
        pt2 = points[p2]
        pt3 = points[p3]
        cv2.line(img, pt1, pt2, (255, 255, 255))
        cv2.line(img, pt2, pt3, (255, 255, 255))
        cv2.line(img, pt3, pt1, (255, 255, 255))
        
    plt.figure(figsize=(12, 9))
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for idx, p in enumerate(points):
        plt.text(p[0], p[1], str(idx),
            bbox={'facecolor':'black', 'alpha':0.7, 'pad':3}, color='white')
    plt.show()

def pt_in_cnt(pt, cnt):
    return cv2.pointPolygonTest(cnt, pt, False) >= 0.0


def emptyImg(img):
    return np.zeros(img.shape, dtype=np.uint8)

# warp to reference face
def force_warping(path):
    cur_points, src = get_face(path)
    ref_points, dst = get_face(REF_FACE_PATH)
    
    for t in REF_MESH:
        p1, p2, p3 = t
        srcTri = np.float32([[cur_points[p1], cur_points[p2], cur_points[p3]]])
        dstTri = np.float32([[ref_points[p1], ref_points[p2], ref_points[p3]]])
        dst = pasteimg(src, dst, srcTri, dstTri)
        
    return dst

def display_face(img, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    plt.show()

PT_FILTER = get_pt_filter()
class AAM:
    def __init__(self, root, ref_id, num_frames):
        self.FACES_PATH = root
        self.REF_FACE_PATH = os.path.join(root, str(ref_id).zfill(3))
        self.REF_FACE_POINTS, self.REF_FACE_IMG = get_face(self.REF_FACE_PATH)
        self.REF_MESH = triangulation(self.REF_FACE_PATH)
        self.NUM_FRAMES = num_frames

        cnt = np.array(self.REF_FACE_POINTS[:11] + [self.REF_FACE_POINTS[15]] + [self.REF_FACE_POINTS[11]])
        cnt_mouth = np.array(self.REF_FACE_POINTS[28:])
        
        self.POINTS_MOUTH = []
        self.POINTS_OUTER = []
        for r in range(self.REF_FACE_IMG.shape[0]):
            for c in range(self.REF_FACE_IMG.shape[1]):
                if pt_in_cnt((c, r), cnt):
                    if pt_in_cnt((c, r), cnt_mouth):
                        self.POINTS_MOUTH.append((c, r))
                    else:
                        self.POINTS_OUTER.append((c, r))

        self.mean_shape = None
        self.mean_mouth = None
        self.mean_outer = None
        self.pca_shape = None
        self.pca_mouth = None
        self.pca_outer = None

        self.Wp = None
        self.Wlambda = None
        self.b = None
        self.pca_b = None
        self.lowdim_b = None
        self.lowdim_b_std = None

    def pasteimg(self, srcimg, dstimg, srcTri, dstTri):
        M = cv2.getAffineTransform(srcTri, dstTri)
        warpped_img = cv2.warpAffine(srcimg, M, (self.REF_FACE_IMG.shape[1], self.REF_FACE_IMG.shape[0]))
        
        for r in range(dstimg.shape[0]):
            for c in range(dstimg.shape[1]):
                if pt_in_cnt((c, r), dstTri[0]):
                    dstimg[r, c] = warpped_img[r, c]

        return dstimg

    def warp_face(self, srcFace, dstFace):
        src_points, src = srcFace
        ref_points, dst = dstFace
        
        for t in self.REF_MESH:
            p1, p2, p3 = t
            src_tri = np.float32([[src_points[p1], src_points[p2], src_points[p3]]])
            dst_tri = np.float32([[ref_points[p1], ref_points[p2], ref_points[p3]]])
            dst = self.pasteimg(src, dst, src_tri, dst_tri)
            
        return dst

    def fit(self):
        mouth = []
        outer = []
        for i in tqdm(range(self.NUM_FRAMES)):
            srcFace = get_face(os.path.join(self.FACES_PATH, str(i+1).zfill(3)))
            dstFace = get_face(self.REF_FACE_PATH)
            dstFace = (dstFace[0], emptyImg(dstFace[1]))
            img = self.warp_face(srcFace, dstFace)

            pixel_mouth = []
            for p in self.POINTS_MOUTH:
                r, c = p
                pixel_mouth.append(img[c, r, :3])
            pixel_mouth = np.float64(pixel_mouth)
            
            pixel_outer = []
            for p in self.POINTS_OUTER:
                r, c = p
                pixel_outer.append(img[c, r, :3])
            pixel_outer = np.float64(pixel_outer)    
                
            mouth.append(pixel_mouth.flatten())
            outer.append(pixel_outer.flatten())

        mouth = np.float64(mouth) / 255.0
        outer = np.float64(outer) / 255.0

        self.mean_mouth = np.mean(mouth, axis=0)
        mouth -= self.mean_mouth

        self.pca_mouth = PCA(n_components=16)
        self.pca_mouth.fit(mouth)
        lowdim_mouth = self.pca_mouth.transform(mouth)
        lowdim_mouth_std = np.std(lowdim_mouth, axis=0)

        self.mean_outer = np.mean(outer, axis=0)
        outer -= self.mean_outer
        self.pca_outer = PCA(n_components=16)
        self.pca_outer.fit(outer)
        lowdim_outer = self.pca_outer.transform(outer)
        lowdim_outer_std = np.std(lowdim_outer, axis=0)

        all_points = []
        for i in range(self.NUM_FRAMES):
            pts = get_points(os.path.join(self.FACES_PATH, str(i+1).zfill(3)) + "_sol")
            CENTER_POINT_IDX = 30
            CENTER_POINT = pts[CENTER_POINT_IDX]

            for idx, p in enumerate(pts):
                pts[idx] = (p[0] - CENTER_POINT[0], p[1] - CENTER_POINT[1])
            pts = np.array(pts)
            all_points.append(pts.flatten())
        all_points = np.float64(all_points)

        self.mean_shape = np.mean(all_points, axis=0)
        all_points -= self.mean_shape

        self.pca_shape = PCA(n_components=16)
        self.pca_shape.fit(all_points)
        lowdim_shape = self.pca_shape.transform(all_points)
        lowdim_shape_std = np.std(lowdim_shape, axis=0)

        self.Wp = np.sqrt(np.sum(lowdim_mouth_std**2) / np.sum(lowdim_shape_std**2))
        self.Wlambda = np.sqrt(np.sum(lowdim_mouth_std**2) / np.sum(lowdim_outer_std**2))
        self.b = np.column_stack((self.Wp * lowdim_shape, self.Wlambda * lowdim_outer, lowdim_mouth))

        self.pca_b = PCA(n_components=16)
        self.pca_b.fit(self.b)
        self.lowdim_b = self.pca_b.transform(self.b)
        self.lowdim_b_std = np.std(self.lowdim_b, axis=0)

    def reconstruct_face(self, test_vector, position=(65, 30)):
        test_b = self.pca_b.inverse_transform(test_vector)
        test_shape = self.mean_shape + self.pca_shape.inverse_transform(test_b[:16] / self.Wp)
        test_shape[::2] += position[0]
        test_shape[1::2] += position[1]
        test_outer = self.mean_outer + self.pca_outer.inverse_transform(test_b[16:32] / self.Wlambda)
        test_mouth = self.mean_mouth + self.pca_mouth.inverse_transform(test_b[32:])
        test_outer *= 255.0
        test_mouth *= 255.0
        
        pic = emptyImg(self.REF_FACE_IMG)
        
        i = 0
        m = test_outer.reshape((-1, 3))
        for p in self.POINTS_OUTER:
            r, c = p
            pic[c, r, :4] = np.int8(np.append(m[i], 255.0))
            i += 1
        
        i = 0
        m = test_mouth.reshape((-1, 3))
        for p in self.POINTS_MOUTH:
            r, c = p
            pic[c, r, :4] = np.int8(np.append(m[i], 255.0))
            i += 1
        
        points = []
        for i in test_shape.reshape(-1, 2):
            points.append((i[0], i[1]))
        
        srcFace = (self.REF_FACE_POINTS, pic)
        dstFace = (points, emptyImg(self.REF_FACE_IMG))
        dst = self.warp_face(srcFace, dstFace)

        return dst
