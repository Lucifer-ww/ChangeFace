# 基于OpenCV的面部特征交换
# 作者: Charles
# 公众号: Charles的皮卡丘
import numpy as np
import dlib
import cv2


# 脸
FACE_POINTS = list(range(17, 68))
# 嘴巴
MOUTH_POINTS = list(range(48, 61))
# 右眉毛
RIGHT_BROW_POINTS = list(range(17, 22))
# 左眉毛
LEFT_BROW_POINTS = list(range(22, 27))
# 右眼
RIGHT_EYE_POINTS = list(range(36, 42))
# 左眼
LEFT_EYE_POINTS = list(range(42, 48))
# 鼻子
NOSE_POINTS = list(range(27, 35))
# 下巴
JAW_POINTS = list(range(0, 17))


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


def GetLandmarks(img_path):
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	rects = detector(img, 1)
	if len(rects) > 1:
		print('[Warning]: More than one face in picture, only choose one randomly...')
		rects = rects[0]
	elif len(rects) == 0:
		print('[Error]: No face detected...')
		return None
	return img, np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


# refer:
# 	https://en.wikipedia.org/wiki/Procrustes_analysis#Ordinary_Procrustes_analysis
def TransferPoints(points1, points2):
	points1 = points1.astype(np.float64)
	points2 = points2.astype(np.float64)
	c1 = np.mean(points1, axis=0)
	c2 = np.mean(points2, axis=0)
	points1 -= c1
	points2 -= c2
	s1 = np.std(points1)
	s2 = np.std(points2)
	points1 /= s1
	points2 /= s2
	# 奇异值分解
	U, S, Vt = np.linalg.svd(points1.T * points2)
	R = (U * Vt).T
	return np.vstack([np.hstack(((s2/s1)*R, c2.T-(s2/s1)*R*c1.T)), np.matrix([0., 0., 1.])])


def DrawConvexHull(img, points, color):
	points = cv2.convexHull(points)
	cv2.fillConvexPoly(img, points, color=color)


def GetFaceMask(img, landmarks):
	img = np.zeros(img.shape[:2], dtype=np.float64)
	groups = [ 
		LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
		NOSE_POINTS + MOUTH_POINTS,
		]
	for group in groups:
		DrawConvexHull(img, landmarks[group], color=1)
	img = np.array([img, img, img]).transpose((1, 2, 0))
	img = (cv2.GaussianBlur(img, (11, 11), 0) > 0) * 1.0
	img = cv2.GaussianBlur(img, (11, 11), 0)
	return img


def WarpImg(img, M, dshape):
	output_img = np.zeros(dshape, dtype=img.dtype)
	cv2.warpAffine(img,
				   M[:2],
				   (dshape[1], dshape[0]),
				   dst=output_img,
				   borderMode=cv2.BORDER_TRANSPARENT,
				   flags=cv2.WARP_INVERSE_MAP)
	return output_img


def ModifyColor(img1, img2, landmarks1):
	blur_amount = 0.6 * np.linalg.norm(np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) - np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
	blur_amount = int(blur_amount)
	if blur_amount % 2 == 0:
		blur_amount += 1
	img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
	img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)
	img2_blur += (128 * (img2_blur <= 1.0)).astype(img2_blur.dtype)
	return (img2.astype(np.float64) * img1_blur.astype(np.float64) / img2_blur.astype(np.float64))


def main(img1, img2):
	ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
	img1, landmarks1 = GetLandmarks(img1)
	img2, landmarks2 = GetLandmarks(img2)
	M = TransferPoints(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
	mask = GetFaceMask(img2, landmarks2)
	warped_mask = WarpImg(mask, M, img1.shape)
	combined_mask = np.max([GetFaceMask(img1, landmarks1), warped_mask], axis=0)
	warped_img2 = WarpImg(img2, M, img1.shape)
	warped_corrected_img2 = ModifyColor(img1, warped_img2, landmarks1)
	output_img = img1 * (1.0 - combined_mask) + warped_corrected_img2 * combined_mask
	cv2.imwrite('output.jpg', output_img)


if __name__ == '__main__':
	main('./1.png', './2.png')