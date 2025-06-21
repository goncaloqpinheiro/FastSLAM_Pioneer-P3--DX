import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy


# ===== Simulation Settings =====
np.random.seed(42)
num_steps = 350 
num_particles = 200
landmarks = np.array([[8, 4], [5, 7], [9, 5], [6, 9], [8, 7]])
sensor_range = 1.8 # Its suposed to be 1.8 by the ArUco paper  
motion_noise = [0.1, 0.05]      # [standard deviation of velocity, standard deviation of angular velocity]
measurement_noise = [0.2, 0.1]  # [standard deviation of range, standard deviation of bearing]
dt = 0.1    # Time step for motion model
# ===== Extra Simulation Settings/Debug =====
resampling = True
noise = True
mode = 1  # 1 = autonomous circle, 2 = manual control
circle_radius = 1.5  # meters
circle_speed = 0.2   # m/s

resample_method = 'low_variance'  # Options: 'low_variance', 'stratified'

print("Select mode:")
print("1 - Autonomous circular motion")
print("2 - Manual control")
mode = int(input("Enter choice (1 or 2): "))

# ===== Autonomous control function =====
def autonomous_control():
    global pose
    # Circular motion control logic
    v = circle_speed
    w = -v / circle_radius  # Angular velocity for circular motion
    return [v, w]


# ===== Helper Functions =====
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Motion model with noise
def motion_model(pose, control):
    v, w = control
    v += np.random.normal(0, motion_noise[0])
    w += np.random.normal(0, motion_noise[1])
    theta = pose[2] + w * dt
    x = pose[0] + v * np.cos(theta) * dt
    y = pose[1] + v * np.sin(theta) * dt
    return np.array([x, y, wrap_angle(theta)])


# Motion model without noise
# This is the expected measurement for a given pose and landmark
# It uses the range and bearing to determine the expected position of the landmark given the robot's pose
# (x,y,theta) is robot pose
# (lx,ly) is landmark position
# The function returns the expected range and bearing to the landmark from the robot's pose
# The range is the distance to the landmark, and the bearing is the angle from the robot's heading to the landmark
def measurement_model(pose, landmark):
    dx = landmark[0] - pose[0]
    dy = landmark[1] - pose[1]
    r = np.hypot(dx, dy)
    bearing = wrap_angle(np.arctan2(dy, dx) - pose[2])
    return np.array([r, bearing])

# In EKF (Extended Kalman Filter), we linearize nonlinear functions using their Jacobian
# This function computes the Jacobian of the measurement model with respect to the landmark position,
# because the measurement model is nonlinear
# The Jacobian is a matrix that describes how the expected measurement changes with small changes in the landmark position
def compute_jacobian(pose, landmark):
    dx = landmark[0] - pose[0]
    dy = landmark[1] - pose[1]
    q = dx**2 + dy**2
    sqrt_q = np.sqrt(q)
    H = np.array([
        [dx / sqrt_q, dy / sqrt_q],
        [-dy / q, dx / q]
    ])
    return H, q

# Used by each particle to update its estimate of a landmark’s position after a measurement
# mu: estimated mean of landmark position
# sigma: estimated covariance of landmark position
# z: measurement (range and bearing)
# pose: robot pose

def ekf_update(mu, sigma, z, pose):
    z_hat = measurement_model(pose, mu)
    H, q = compute_jacobian(pose, mu)
    R = np.diag(measurement_noise)**2
    S = H @ sigma @ H.T + R
    K = sigma @ H.T @ np.linalg.inv(S)
    innovation = z - z_hat
    innovation[1] = wrap_angle(innovation[1])
    if abs(innovation[1]) > np.radians(25):     # Changed from 60 to 25, improved some how
    # Skip update — too much angular disagreement
        return mu, sigma, 1e-10
    
    # Calculate Mahalanobis distance of the innovation
    mahalanobis = innovation.T @ np.linalg.inv(S) @ innovation
    if mahalanobis > 9.0:  # ~3 standard deviations
        return mu, sigma, 1e-10  # reject outlier update

    mu = mu + K @ innovation
    sigma = (np.eye(2) - K @ H) @ sigma
    prob = np.exp(-0.5 * innovation.T @ np.linalg.inv(S) @ innovation) \
           / np.sqrt((2 * np.pi)**2 * np.linalg.det(S))
    return mu, sigma, prob


def normalize_weights(particles):
    """Normalize particle weights"""
    weights = np.array([p['weight'] for p in particles])
    weights += 1e-300
    total_weight = np.sum(weights)
    
    if total_weight == 0:
        for p in particles:
            p['weight'] = 1.0 / len(particles)
    else:
        for i, p in enumerate(particles):
            p['weight'] = weights[i] / total_weight
    return particles

def resample(particles, resample_method):
    """Enhanced resampling function with effective particle number check"""
    
    # Normalize weights
    particles = normalize_weights(particles)
    weights = np.array([p['weight'] for p in particles])
    
    # Find highest weight particle index before resampling
    highest_weight_index = np.argmax(weights)
    
    # Calculate effective particle number
    Neff = 1.0 / np.sum(np.square(weights)) if np.sum(np.square(weights)) > 0 else 0
    equal_weights = np.full_like(weights, 1.0 / len(particles))
    Neff_maximum = 1.0 / np.sum(np.square(equal_weights))
    
    new_highest_weight_index = highest_weight_index
    
    # Resample if Neff is too low - particles are not representative of posterior
    if Neff < Neff_maximum / 2:
        #rospy.logdebug(f"Resampling triggered: Neff={Neff:.2f} < {Neff_maximum/2:.2f}")
        
        if resample_method == "low variance":
            indices = low_variance_resampling(weights, equal_weights, len(particles))
        elif resample_method == "Stratified":
            indices = stratified_resampling(weights, len(particles))
        else:
            # Fallback to simple random resampling
            indices = np.random.choice(len(particles), size=len(particles), p=weights)
        
        # Create new particle set
        particles_copy = copy.deepcopy(particles)
        for i in range(len(indices)):
            particles[i]['pose'] = particles_copy[indices[i]]['pose'].copy()
            particles[i]['map'] = copy.deepcopy(particles_copy[indices[i]]['map'])
            particles[i]['weight'] = 1.0 / len(particles)  # Reset weights
            
            # Track new index of highest weight particle
            if highest_weight_index == indices[i]:
                new_highest_weight_index = i
    
    return particles, new_highest_weight_index

#def compute_neff(particles):
#    weights = np.array([p['weight'] for p in particles])
#    weights += 1e-300 # Avoid division by zero
#    weights /= np.sum(weights)
#    return 1.0 / np.sum(np.square(weights))

def low_variance_resampling(weights, equal_weights, num_particles):
    """Low variance resampling algorithm"""
    indices = []
    r = np.random.uniform(0, 1.0/num_particles)
    c = weights[0]
    i = 0
    
    for m in range(num_particles):
        u = r + m * (1.0/num_particles)
        while u > c:
            i += 1
            if i >= len(weights):
                i = len(weights) - 1
                break
            c += weights[i]
        indices.append(i)
    
    return indices

def stratified_resampling(weights, num_particles):
    """Stratified resampling algorithm"""
    indices = []
    cumsum = np.cumsum(weights)
    
    for i in range(num_particles):
        # Generate random number in stratum
        u = (i + np.random.uniform(0, 1)) / num_particles
        # Find corresponding index
        index = np.searchsorted(cumsum, u)
        indices.append(min(index, len(weights) - 1))
    
    return indices

def motion_model_no_noise(pose, control):
    v, w = control
    
    v += np.random.normal(0, motion_noise[0])
    w += np.random.normal(0, motion_noise[1])
    
    theta = pose[2] + w * dt
    x = pose[0] + v * np.cos(theta) * dt
    y = pose[1] + v * np.sin(theta) * dt
    return np.array([x, y, wrap_angle(theta)])



# ===== MAIN FUNCTION =====
# ===== Initialization =====
# ===== Modify the initialization =====
if mode == 1:
    pose = np.array([5.0, 5.0, np.pi/2])  # Start at center facing "north"
else:
    pose = np.array([3.0, 2.0, np.pi/2])  # Original starting position

true_trajectory = [pose.copy()]
estimated_trajectory = []

particles = [{
    'pose': pose.copy() + np.random.normal(0, 0.01, size=pose.shape),  # Add small noise to initial pose
    'map': {},
    'weight': 1.0
} for _ in range(num_particles)]

highest_weight_particle_index = 0

# ===== Plot Setup =====
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("WASD Control: W=Forward, S=Backward, A=Left, D=Right")
ax.grid(True)
landmark_plot = ax.scatter(*zip(*landmarks), c='red', label='True Landmarks')

for i, (lx, ly) in enumerate(landmarks):
    ax.text(lx, ly + 0.3, f"ID {i}", color='black', fontsize=9, ha='center')

true_traj_line, = ax.plot([], [], 'b-', label='True Trajectory')
est_traj_line, = ax.plot([], [], 'g--', label='Estimated Trajectory')
robot_dot = ax.scatter([], [], c='blue', s=80, label='True Robot')
particle_plot = ax.scatter([], [], c='gray', s=10, alpha=0.5, label='Particles')
observed_landmarks = ax.scatter([], [], c='green', s=50, marker='x', label='Observed by Best Particle')
plt.legend()

# Initialize FOV lines (initially empty)
fov_line1, = ax.plot([], [], 'm-', alpha=0.5, linewidth=1, label='FOV Boundary')
fov_line2, = ax.plot([], [], 'm-', alpha=0.5, linewidth=1)
plt.legend()

# ===== Interactive Control Variables =====
current_control = [0.0, 0.0]
target_angle = None
turn_increment = np.radians(90)  # 45 degrees in radians

def on_key(event):
    global current_control, target_angle
    v_step = 0.2
    
    if event.key == 'w' or event.key == 'up':
        current_control[0] = v_step
        current_control[1] = 0  # Stop turning when moving forward
        target_angle = None
    elif event.key == 's' or event.key == 'down':
        current_control[0] = -v_step
        current_control[1] = 0  # Stop turning when moving backward
        target_angle = None
    elif event.key == 'a' or event.key == 'left':
        # Set target angle 45 degrees left
        target_angle = wrap_angle(pose[2] + turn_increment)
    elif event.key == 'd' or event.key == 'right':
        # Set target angle 45 degrees right
        target_angle = wrap_angle(pose[2] - turn_increment)
    elif event.key == ' ':
        current_control = [0.0, 0.0]
        target_angle = None


# ===== Update Function =====
def update(frame):
    global pose, particles, current_control, target_angle
    
    # Handle automatic turning
    if target_angle is not None:
        angle_diff = wrap_angle(target_angle - pose[2])
        if abs(angle_diff) > 0.1:  # If not yet aligned
            current_control[1] = np.sign(angle_diff) * 0.2  # Turn toward target
        else:
            current_control[1] = 0  # Stop turning when aligned
            target_angle = None

    # Get control based on mode
    if mode == 1:
        control = autonomous_control()
    else:
        control = current_control
    
    pose = motion_model_no_noise(pose, control)
    true_trajectory.append(pose.copy())

    for p in particles:
        p['pose'] = motion_model(p['pose'], control)
        for idx, landmark in enumerate(landmarks):
            dx = landmark[0] - p['pose'][0]
            dy = landmark[1] - p['pose'][1]
            distance = np.hypot(dx, dy)
            bearing = wrap_angle(np.arctan2(dy, dx) - p['pose'][2])
            in_fov = -np.radians(28.5) <= bearing <= np.radians(28.5)
            in_range = 0 <= distance <= sensor_range
            #in_fov = True
            #in_range = True

            if in_fov and in_range:
                z = measurement_model(p['pose'], landmark)
                if noise == True:
                    z[0] += np.random.normal(0, measurement_noise[0])
                    z[1] += np.random.normal(0, measurement_noise[1])
                if idx not in p['map']:
                    r, b = z
                    lx = p['pose'][0] + r * np.cos(b + p['pose'][2])
                    ly = p['pose'][1] + r * np.sin(b + p['pose'][2])
                    p['map'][idx] = {'mu': np.array([lx, ly]), 'sigma': np.eye(2) * 1.0}
                else:
                    mu, sigma = p['map'][idx]['mu'], p['map'][idx]['sigma']
                    mu, sigma, prob = ekf_update(mu, sigma, z, p['pose'])
                    p['map'][idx]['mu'] = mu
                    p['map'][idx]['sigma'] = sigma
                    p['weight'] *= prob

    # Enhanced resampling with effective particle number check
    particles, highest_weight_particle_index = resample(particles, resample_method)
    

    #best_particle = max(particles, key=lambda p: p['weight'])
    best_particle = particles[highest_weight_particle_index]
    estimated_trajectory.append(best_particle['pose'].copy())

    true_x, true_y = zip(*[(p[0], p[1]) for p in true_trajectory])
    est_x, est_y = zip(*[(p[0], p[1]) for p in estimated_trajectory])
    true_traj_line.set_data(true_x, true_y)
    est_traj_line.set_data(est_x, est_y)
    robot_dot.set_offsets([pose[:2]])
    particle_plot.set_offsets(np.array([p['pose'][:2] for p in particles]))

    visible_landmarks = [lm['mu'] for lm in best_particle['map'].values()
                         if np.linalg.norm(lm['mu'] - best_particle['pose'][:2]) < sensor_range]
    if visible_landmarks:
        observed_landmarks.set_offsets(np.array(visible_landmarks))
    else:
        observed_landmarks.set_offsets(np.empty((0, 2)))

# === Clear previous ellipses ===
    [child.remove() for child in ax.patches[:]]
    # Clear previous text labels
    for label in reversed(ax.texts):
        label.remove()


# === Draw uncertainty ellipses for each landmark ===
    for lm in best_particle['map'].values():
        mu = lm['mu']
        sigma = lm['sigma']
            
        # Eigen decomposition to get ellipse shape
        vals, vecs = np.linalg.eigh(sigma)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 6 * np.sqrt(vals)  # 1-sigma ellipse

        ellipse = plt.matplotlib.patches.Ellipse(mu, width, height, angle=angle,
                                                edgecolor='green', facecolor='none', lw=1.5, alpha=0.7)
        ax.add_patch(ellipse)

        # Draw landmark ID below the landmark
        #ax.text(mu[0], mu[1] - 0.2, f"id: {lm_id}", color='black', fontsize=9, ha='center', va='top')
    

    # Update FOV lines
    fov_length = sensor_range  # Length of FOV lines
    x, y, theta = pose
    angle1 = theta + np.radians(28.5)  # Right FOV boundary
    angle2 = theta - np.radians(28.5)  # Left FOV boundary

    # Calculate endpoints of FOV lines
    x1 = x + fov_length * np.cos(angle1)
    y1 = y + fov_length * np.sin(angle1)
    x2 = x + fov_length * np.cos(angle2)
    y2 = y + fov_length * np.sin(angle2)

    fov_line1.set_data([x, x1], [y, y1])
    fov_line2.set_data([x, x2], [y, y2])

    return true_traj_line, est_traj_line, robot_dot, particle_plot, observed_landmarks, fov_line1, fov_line2

# Register key press event
fig.canvas.mpl_connect('key_press_event', on_key)

# Run animation
import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=100, repeat=False)
plt.show()

def plot_final_map(best_particle, true_landmarks):
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.set_title("Final Map Estimate (Best Particle)")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.grid(True)

    # Plot true landmarks
    ax2.scatter(*zip(*true_landmarks), c='red', marker='o', label='True Landmarks')

    # Plot estimated landmarks with uncertainty ellipses
    for idx, lm in best_particle['map'].items():
        mu = lm['mu']
        sigma = lm['sigma']

        # Plot estimated position
        ax2.scatter(mu[0], mu[1], c='green', marker='x', label='Estimated Landmark' if idx == 0 else "")

        # Plot uncertainty ellipse
        vals, vecs = np.linalg.eigh(sigma)
        angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        #width, height = 2 * np.sqrt(vals)
        width, height = 6 * np.sqrt(vals)  # 3-sigma ellipse (covers 99.7% of the uncertainty)
        ellipse = plt.matplotlib.patches.Ellipse(mu, width, height, angle=angle, alpha=0.3, color='green')
        ax2.add_patch(ellipse)

        # Label landmarks
        ax2.text(mu[0], mu[1] + 0.2, f"ID {idx}", color='green', fontsize=8, ha='center')

    ax2.legend()
    plt.tight_layout()
    plt.savefig("final_map_output.png")
    plt.show()

# After animation ends, plot the final map
best_particle = particles[highest_weight_particle_index]
#best_particle = max(particles, key=lambda p: p['weight'])
plot_final_map(best_particle, landmarks)

print("\n=== Final Map Estimate from Best Particle ===")
total_error = 0
for idx, lm in best_particle['map'].items():
    mu = lm['mu']
    true_pos = landmarks[idx]
    error = np.linalg.norm(mu - true_pos)
    total_error += error
    print(f"Landmark {idx}:")
    print(f"  → Estimated: [{mu[0]:.2f}, {mu[1]:.2f}]  → True     : [{true_pos[0]:.2f}, {true_pos[1]:.2f}]  → Error    : {error:.2f} m")




