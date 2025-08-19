import requests
import json
import base64
import time
import numpy as np
import cv2
import asyncio
import websockets

# --- Configuration ---
FLASK_URL = "http://localhost:5000"
WEBSOCKET_URL = "ws://localhost:8080"
# Set the goal to the South-West (SW) corner, which is (-45, -45)
GOAL_POSITION = {"x": -35, "z": -35}
# Distance threshold to consider the goal reached
GOAL_THRESHOLD = 5

# --- Image Analysis ---
def analyze_image(image_data_url):
    """
    Analyzes the robot camera image to detect green obstacles.
    Returns True if an obstacle is likely present, False otherwise.
    """
    try:
        # Decode the base64 image data from the data URL
        header, encoded = image_data_url.split(",", 1)
        data = base64.b64decode(encoded)

        # Convert the raw bytes to a format readable by OpenCV
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert from BGR to HSV color space for better green detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the green color of the obstacles
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        # Create a mask to isolate green pixels
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Count the number of green pixels
        green_pixel_count = np.sum(mask > 0)
        total_pixels = img.shape[0] * img.shape[1]
        green_percentage = (green_pixel_count / total_pixels) * 100

        # If the percentage of green pixels is above a small threshold, an obstacle is detected
        return green_percentage > 0.1
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return False

# --- Robot Control Logic ---
async def capture_image_from_server(ws):
    """
    Sends a capture command to the server and waits for the image
    response from the simulator via WebSocket.
    """
    requests.post(f"{FLASK_URL}/capture")
    try:
        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(message)
            if data.get("type") == "capture_image_response":
                return data
    except asyncio.TimeoutError:
        print("Timeout waiting for image response.")
        return None

async def main():
    print("🤖 Autonomous Robot Controller started.")
    print(f"🎯 Goal set to: {GOAL_POSITION}")

    # Establish the WebSocket connection
    async with websockets.connect(WEBSOCKET_URL) as ws:
        # First, set the goal position in the simulator
        requests.post(f"{FLASK_URL}/goal", json=GOAL_POSITION)
        print("Goal position sent to simulator.")

        # Main control loop
        while True:
            # Step 1: Perception - Capture a camera image
            image_response = await capture_image_from_server(ws)
            if not image_response:
                print("Failed to get image. Retrying...")
                await asyncio.sleep(1)
                continue

            image_data = image_response["image"]
            robot_pos = image_response["position"]

            # Step 2: Check for Goal Proximity
            distance_to_goal = ((robot_pos["x"] - GOAL_POSITION["x"])**2 + (robot_pos["z"] - GOAL_POSITION["z"])**2)**0.5
            if distance_to_goal < GOAL_THRESHOLD:
                print("🎉 Goal reached! Stopping.")
                requests.post(f"{FLASK_URL}/stop")
                break

            # Step 3: Decision - Analyze the image for obstacles
            is_obstacle_ahead = analyze_image(image_data)

            # Step 4: Action - Move based on the analysis
            if is_obstacle_ahead:
                print("⚠️ Obstacle detected! Turning and moving to clear...")
                # Turn right by a small angle and move forward a bit
                requests.post(f"{FLASK_URL}/move_rel", json={"turn": 20, "distance": 5})
                await asyncio.sleep(2)  # Pause to allow the turn and move to complete
            else:
                print("✅ Path clear. Moving forward a step...")
                # Move forward a fixed, small step towards the goal
                requests.post(f"{FLASK_URL}/move_rel", json={"turn": 0, "distance": 10})
                await asyncio.sleep(1) # Pause to allow the step to complete
                # Re-orient towards the goal
                requests.post(f"{FLASK_URL}/move", json=GOAL_POSITION)


            # Wait a moment before the next cycle to prevent command spamming
            await asyncio.sleep(1)

# Run the main asynchronous function
if __name__ == "__main__":
    asyncio.run(main())