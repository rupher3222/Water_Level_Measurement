import cv2
import numpy as np

# Global variables
roi_corners = []  # [(x1, y1), (x2, y2)]
projection_points = []  # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
original_image = None
mask_image = None
roi_defined = False
projection_points_defined = False

def mouse_callback(event, x, y, flags, param):
    global roi_corners, projection_points, original_image, roi_defined, projection_points_defined
    if event == cv2.EVENT_LBUTTONDOWN:
        if not roi_defined:  # Selecting ROI corners
            if len(roi_corners) < 2:
                roi_corners.append((x, y))
                cv2.circle(original_image, (x, y), 3, (0, 255, 0), -1)
                if len(roi_corners) == 2:
                    cv2.rectangle(original_image, roi_corners[0], roi_corners[1], (0, 255, 0), 2)
        elif roi_defined and len(projection_points) < 4:  # Selecting projection points
            projection_points.append((x, y))
            cv2.circle(original_image, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Original Image", original_image)

def setup_user_input(original_path, mask_path):
    global original_image, mask_image, roi_defined, projection_points_defined
    original_image = cv2.imread(original_path)
    mask_image = cv2.imread(mask_path, 0)

    # Resize original image to match the mask image size
    original_image = cv2.resize(original_image, (mask_image.shape[1], mask_image.shape[0]))

    cv2.namedWindow("Original Image")
    cv2.setMouseCallback("Original Image", mouse_callback)
    cv2.imshow("Original Image", original_image)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter key
            if not roi_defined and len(roi_corners) == 2:
                roi_defined = True
                print("ROI defined, now select 4 projection points.")
            elif roi_defined and not projection_points_defined and len(projection_points) == 4:
                projection_points_defined = True
                print("Projection points defined, processing transformation.")
                break

    cv2.destroyAllWindows()
    process_transformation()

def process_transformation():
    global original_image, mask_image, roi_corners, projection_points
    x1, y1 = roi_corners[0]
    x2, y2 = roi_corners[1]
    roi_original = original_image[y1:y2, x1:x2]
    roi_mask = mask_image[y1:y2, x1:x2]

    # 사용자가 ROI 영역을 선택한 후 마스크 이미지의 해당 부분을 미리 보기
    cv2.imshow("ROI Mask Preview", roi_mask)  # ROI 마스크 미리보기
    cv2.waitKey(0)  # 사용자 입력을 기다림

    src_pts = np.array(projection_points, dtype="float32")
    dst_pts = np.array([[0, 0], [roi_mask.shape[1]-1, 0], [0, roi_mask.shape[0]-1], [roi_mask.shape[1]-1, roi_mask.shape[0]-1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    transformed_original = cv2.warpPerspective(roi_original, matrix, (roi_mask.shape[1], roi_mask.shape[0]))
    transformed_mask = cv2.warpPerspective(roi_mask, matrix, (roi_mask.shape[1], roi_mask.shape[0]))

    cv2.imshow("Transformed Original", transformed_original)
    cv2.imshow("Transformed Mask", transformed_mask)
    cv2.waitKey(0)

    water_level = detect_water_level(transformed_mask)
    if water_level is not None:
        print("Detected Water Level:", water_level)
    else:
        print("Failed to detect water level.")
    cv2.destroyAllWindows()

def detect_water_level(mask):
    edges = cv2.Canny(mask, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        y_coords = [line[0][1] for line in lines]
        average_y = np.mean(y_coords)
        scale = -0.01
        offset = 500
        water_level = average_y * scale + offset
        return water_level
    return None

if __name__ == "__main__":
    original_path = 'D:\DEV_research\WATER-LEVEL\src\original\original.png'
    mask_path = 'D:\DEV_research\WATER-LEVEL\src\mask\mask.png'
    setup_user_input(original_path, mask_path)
