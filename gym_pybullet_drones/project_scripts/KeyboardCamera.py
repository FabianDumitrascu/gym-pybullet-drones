import pybullet as p
import pybullet_data
import time

class KeyboardControlledCamera:
    """Class to control the PyBullet camera using keyboard keys (WASD for rotation, Q/E for zoom)."""
    def __init__(self, client_id, update_interval=0.1):
        self.client_id = client_id
        self.camera_distance = 5.0  # Initial zoom distance
        self.camera_yaw = 45
        self.camera_pitch = -30
        self.camera_target = [0, 0, 1]
        self.last_update_time = time.time()
        self.update_interval = update_interval  # Time interval between updates (seconds)

    def update_camera_with_keyboard(self):
        """
        Update the camera based on keyboard input:
        W/S: Pitch up/down
        A/D: Yaw left/right
        Q/E: Zoom in/out
        """
        # Throttle updates to make it less responsive
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return  # Skip updates until the interval has passed

        keys = p.getKeyboardEvents(physicsClientId=self.client_id)

        # Camera adjustment step
        YAW_STEP = 5
        PITCH_STEP = 5
        ZOOM_STEP = 0.5  # Zoom increment (distance change)

        # WASD keys for yaw and pitch
        if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
            self.camera_pitch -= PITCH_STEP

        if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
            self.camera_pitch += PITCH_STEP

        if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
            self.camera_yaw -= YAW_STEP

        if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
            self.camera_yaw += YAW_STEP

        # Q and E keys for zooming
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
            self.camera_distance += ZOOM_STEP  # Zoom out

        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
            self.camera_distance = max(1.0, self.camera_distance - ZOOM_STEP)  # Zoom in (minimum distance = 1)

        # Clamp pitch to avoid flipping
        self.camera_pitch = max(-89, min(89, self.camera_pitch))

        # Update the camera
        p.resetDebugVisualizerCamera(
            self.camera_distance,
            self.camera_yaw,
            self.camera_pitch,
            self.camera_target,
            physicsClientId=self.client_id
        )

        # Update the last update time
        self.last_update_time = current_time



def main():
    """Main function to test keyboard-controlled camera."""
    # Connect to PyBullet GUI
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load a plane and an object
    p.loadURDF("plane.urdf", physicsClientId=client_id)
    p.loadURDF("r2d2.urdf", [0, 0, 1], physicsClientId=client_id)

    # Enable keyboard input and GUI
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)

    print("Use WASD to control the camera (W: up, S: down, A: left, D: right).")

    # Initialize the keyboard-controlled camera
    camera = KeyboardControlledCamera(client_id)

    # Simulation loop
    while True:
        p.stepSimulation()
        camera.update_camera_with_keyboard()  # Update the camera based on keyboard input
        time.sleep(1 / 240)  # Match simulation frequency

if __name__ == "__main__":
    main()
