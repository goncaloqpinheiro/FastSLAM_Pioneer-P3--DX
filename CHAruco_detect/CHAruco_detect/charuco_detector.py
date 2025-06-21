import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# === Load calibration data ===
data = np.load("calibration_data_charuco.npz")
camera_matrix = data['camera_matrix']
dist_coeffs = data['dist_coeffs']

# === Marker setup ===
marker_length = 0.15  # 1.5 cm
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters_create()

# === Function: Convert rotation vector to Euler angles ===
def get_euler_angles(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])  # pitch, yaw, roll

# === Function: Draw cube on marker ===
def draw_cube(frame, rvec, tvec, camera_matrix, dist_coeffs):
    cube_points = np.float32([
        [0, 0, 0], [marker_length, 0, 0],
        [marker_length, marker_length, 0], [0, marker_length, 0],
        [0, 0, -marker_length], [marker_length, 0, -marker_length],
        [marker_length, marker_length, -marker_length], [0, marker_length, -marker_length]
    ])
    imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0,255,0), 2)
    for i in range(4):
        frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[i+4]), (255,0,0), 2)
    frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0,0,255), 2)
    return frame

# === Start video capture ===
cap = cv2.VideoCapture(0)
marker_positions = []

frame_count = 0
max_frames = 600  # 150frames~5 seconds at 30 FPS

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            rvec, tvec = rvecs[i], tvecs[i]
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.01)
            frame = draw_cube(frame, rvec, tvec, camera_matrix, dist_coeffs)

            # Save marker position for 2D plotting
            marker_positions.append(tvec[0])

            # Distance and orientation
            distance = np.linalg.norm(tvec)
            pitch, yaw, roll = get_euler_angles(rvec)
            c = tuple(np.int32(corners[i][0][0]))
            text = f"ID:{ids[i][0]} | Dist:{distance*100:.1f}cm | Pitch:{pitch:.1f}° Yaw:{yaw:.1f}° Roll:{roll:.1f}°"
            cv2.putText(frame, text, (c[0], c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Pose Estimation", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

# === 2D Plot: Top-Down View (X vs Z) ===
if marker_positions:
    positions = np.array(marker_positions)

    plt.figure(figsize=(8, 6))
    plt.scatter(positions[:, 0], positions[:, 2], c='blue', label='Marker Position')

    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel('X (meters) – left/right')
    plt.ylabel('Z (meters) – forward')
    plt.title('2D Marker Position Relative to Camera (Top-Down View)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
else:
    print("No marker positions recorded.")

