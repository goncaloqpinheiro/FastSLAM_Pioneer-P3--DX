import cv2
import numpy as np
import os

# === Setup: Board Parameters (based on your measurements) ===
squares_x = 5
squares_y = 7
square_length = 0.03   # in meters (3 cm)
marker_length = 0.015  # in meters (1.5 cm)

# Create ChArUco board
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, aruco_dict)

# Create folder to save images
save_dir = 'charuco_calib_images'
os.makedirs(save_dir, exist_ok=True)

# === Step 1: Automatically Capture 30 Good Frames ===
cap = cv2.VideoCapture(0)
print("[INFO] Starting capture... looking for good ChArUco detections.")
img_counter = 0
target_images = 30

while img_counter < target_images:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if len(corners) > 0:
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if retval and retval > 10:
            fname = os.path.join(save_dir, f"charuco_{img_counter:02d}.png")
            cv2.imwrite(fname, frame)
            print(f"[INFO] Captured {img_counter + 1}/{target_images} good frames")
            img_counter += 1
            cv2.waitKey(500)  # short pause to avoid duplicates

    # Display preview
    cv2.imshow("ChArUco Capture", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# === Step 2: Calibrate Camera ===
print("[INFO] Starting calibration...")

all_corners = []
all_ids = []
image_size = None
image_files = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith(".png")]

for fname in image_files:
    image = cv2.imread(fname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if len(corners) > 0:
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if retval and retval > 10:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)

if len(all_corners) == 0:
    print("[ERROR] No valid ChArUco corners found. Check your images and try again.")
else:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print("[INFO] Calibration successful.")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    np.savez("calibration_data_charuco.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("[INFO] Calibration data saved to 'calibration_data_charuco.npz'")
