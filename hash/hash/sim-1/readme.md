# Robot Simulator Project
## Overview
This project integrates a Three.js-based robot simulator with a Python WebSocket server and Flask API endpoints. The simulator supports both absolute and relative movement commands, while collision detection is handled on the client side (in the Three.js code).

## Behavior & Notes

* The floor is **100 × 100** centered at the origin, so corners are approximately `(±50, 0, ±50)`.
  The goal is placed slightly **inset** (default margin ≈ 5 units) to keep it inside the floor bounds.
  Example effective corners: `(±45, 0, ±45)` with signs depending on NE/NW/SE/SW.
* Setting a new goal **repositions** the existing goal post (no delete endpoint needed).
* The goal is purely a **visual/notification target**; it does **not** drive the robot. Use `/move` or `/move_rel` to navigate.


## Prerequisites
- **Python 3.7+**
- **pip** (Python package installer)
- A modern web browser (e.g., Chrome, Firefox) to run the Three.js simulator
- (Optional) An HTTP client (e.g., curl, Postman) for testing the Flask API endpoints

## Required Python Packages
Install the following packages using pip:
```bash
pip install websockets flask
```

## Setup and Running the Project 
- git clone git@github.com:terafac/sim-1.git
- cd sim-1
- Install Dependencies: Open a terminal in the project directory and run:
```bash
pip install websockets flask
``` 
- Run the Python Server:
- In the terminal, navigate to the project directory.

- Start the server by running:
```bash
python server.py
```
The server will start:

A WebSocket server on ws://localhost:8080

A Flask API on port 5000

Open the Three.js Simulator:

Open the index.html file in your web browser.

To run from terminal on port 8000, you can use the command

```bash
python -m http.server
```

The simulator will automatically connect to the WebSocket server at ws://localhost:8080.

Check your browser’s console or the server terminal for connection messages (e.g., "Client connected via WebSocket").

# API Endpoints and Simulator Responses

You can control the robot by sending HTTP POST requests to the Flask API endpoints.

---

## 1. Absolute Movement

- **Endpoint:** `/move`
- **Method:** `POST`

### Payload Example:
```json
{
  "x": 10,
  "z": -5
}

```
## Curl Example:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"x": 10, "z": -5}' http://localhost:5000/move
```
## 2. Relative Movement


- **Endpoint:** `/move_rel`
- **Method:** `POST`

### Payload Example:
```json
{
  "turn": 45,
  "distance": 10
}

```
## Curl Example:
```bash 
curl -X POST -H "Content-Type: application/json" -d '{"turn": 45, "distance": 10}' http://localhost:5000/move_rel
```

## 3.Stop Command


- **Endpoint:** `/stop`
- **Method:** `POST`

## Curl Example:
```bash 
curl -X POST http://localhost:5000/stop
```
## 4. Capture Image


- **Endpoint:** `/capture`
- **Method:** `POST`

## Curl Example 
```bask 
curl -X POST http://localhost:5000/capture
```

# Simulator Responses
The Three.js simulator sends back WebSocket messages. Examples include:

## 1. Absolute Move Confirmation:
```bask 
{
  "type": "confirmation",
  "message": "Move command received",
  "target": { "x": 10, "y": 0, "z": -5 }
}
```
## 2. Relative Move Confirmation:
```bash
{
  "type": "confirmation",
  "message": "Relative move command executed",
  "target": { "angle": 45, "distance": 10 }
}
```
## 3. Collision Notification
```bash
{
  "type": "collision",
  "collision": true,
  "position": { "x": <current_x>, "y": <current_y>, "z": <current_z> },
  "obstacle": { "position": { "x": <obs_x>, "y": <obs_y>, "z": <obs_z> } }
}
```

## 4. Capture Image Response:
```bash 
{
  "type": "capture_image_response",
  "image": "<Base64 encoded PNG image>",
  "timestamp": <timestamp>,
  "position": { "x": <current_x>, "y": <current_y>, "z": <current_z> }
}
```

Here’s a plug-and-play section you can drop into your README to document the **Goal Post** feature.


## 5. Goal Post (Set a target corner or exact coordinates)

The simulator supports a visible **goal post** (flag + pole). You can set it to any of the floor’s four corners or to explicit coordinates. When the robot reaches the goal’s bounding box, the simulator emits a `goal_reached` WebSocket message.

> **Requires:** the updated `server.py` and `index.html` from this repo (adds `/goal` endpoint and client-side goal rendering).

### Endpoint

* **URL:** `POST /goal`
* **Content-Type:** `application/json`

You can specify either a **corner** or **coordinates**:

#### Option A — Corner

Allowed values: `NE`, `NW`, `SE`, `SW` (aliases supported: `TR/TL/BR/BL`).

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"corner":"NE"}' \
  http://localhost:5000/goal
```

#### Option B — Coordinates

Provide `x` and `z` (y is optional, defaults to 0).

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"x": 45, "z": -45}' \
  http://localhost:5000/goal
```


### Example Responses

**Flask API (HTTP)**

```json
{
  "status": "goal set",
  "goal": { "x": 45, "y": 0, "z": -45 }
}
```

**Simulator (WebSocket → your browser console / control UI)**

```json
{ "type": "confirmation", "message": "Goal set", "position": { "x": 45, "y": 0, "z": -45 } }
```

When the robot reaches the goal:

```json
{
  "type": "goal_reached",
  "position": { "x": 44.9, "y": 0, "z": -45.1 }
}
```

### Troubleshooting

* **No goal appears:** make sure your browser is running the updated `index.html` and it’s connected to the WebSocket (`ws://localhost:8080`).
* **No WebSocket messages:** ensure the simulator page is open and the server is running. Check terminal logs for “WebSocket connected”.
* **CORS issues (if using a separate control page):** enable the provided CORS snippet in `server.py` or serve your control page from the same origin.

Here’s a drop-in README section you can paste under **API Endpoints and Simulator Responses**. It documents the on-screen collision counter, plus the new **/collisions** and **/reset** endpoints with examples.

---

## 6. Collision Counter & Reset

### What it is

* The simulator shows a **Collisions** counter on-screen (bottom-left HUD).
* The server keeps a **global collision count** (for this server process) by listening to WebSocket messages of type `"collision"` from the simulator.
* You can **query** the current count and **reset** both the server count and the simulator HUD/pose.

### How it works

* Whenever the simulator detects an obstacle collision, it:

  1. stops the robot and changes body color,
  2. increments the **on-screen** counter,
  3. sends a WebSocket message to the server:

     ```json
     {
       "type": "collision",
       "collision": true,
       "position": { "x": <x>, "y": <y>, "z": <z> },
       "obstacle": { "position": { "x": <ox>, "y": <oy>, "z": <oz> } }
     }
     ```
* The server increments its **global** `collision_count` whenever it receives such a message.

> Note: If you have multiple simulators connected, the server’s count is an aggregate of all of them for this server session.


### Get Collision Count

* **Endpoint:** `/collisions`
* **Method:** `GET`
* **Response:**

  ```json
  { "count": 3 }
  ```

**Curl Example**

```bash
curl http://localhost:5000/collisions
```


## 7. Moving Obstacles

The simulator now supports **dynamic obstacles**. You can enable/disable motion, adjust speed, set bounding limits, and optionally provide explicit velocity vectors for each obstacle. When enabled, obstacles update their positions each frame, and collision detection continues to work against the moving boxes.

### Endpoint

* **URL:** `POST /obstacles/motion`
* **Content-Type:** `application/json`

### Payload Options

| Field        | Type   | Description                                                                                                                            |
| ------------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`    | bool   | **Required.** `true` to start motion, `false` to stop.                                                                                 |
| `speed`      | number | Optional. Multiplier applied per frame (default `0.05`).                                                                               |
| `bounds`     | object | Optional. Floor boundaries. Format: `{"minX":-45,"maxX":45,"minZ":-45,"maxZ":45}`                                                      |
| `bounce`     | bool   | Optional. If true, obstacles **bounce** off bounds; if false, they **wrap around**.                                                    |
| `velocities` | array  | Optional. List of `{x, z}` velocity vectors. If omitted, random vectors are generated. Must match the number of obstacles if provided. |

### Example Commands

Enable random-motion obstacles with bounce:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"enabled":true, "speed":0.08, "bounds":{"minX":-45,"maxX":45,"minZ":-45,"maxZ":45}, "bounce":true}' \
  http://localhost:5000/obstacles/motion
```

Enable with explicit velocities (assuming 3 obstacles):

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"enabled":true, "speed":0.06,
       "velocities":[{"x":1,"z":0},{"x":0.5,"z":0.3},{"x":-0.2,"z":-0.4}]}' \
  http://localhost:5000/obstacles/motion
```

Disable obstacle motion:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"enabled":false}' \
  http://localhost:5000/obstacles/motion
```

### Responses

**Flask API (HTTP)**

```json
{
  "status": "obstacle motion updated",
  "config": {
    "command": "set_obstacle_motion",
    "enabled": true,
    "speed": 0.08,
    "bounds": { "minX": -45, "maxX": 45, "minZ": -45, "maxZ": 45 },
    "bounce": true
  }
}
```

**Simulator (WebSocket)**

```json
{ "type": "confirmation", "message": "Obstacle motion updated", "enabled": true }
```

### Behavior Notes

* Moving obstacles **still trigger collision events** if the robot intersects them:

  ```json
  {
    "type": "collision",
    "collision": true,
    "position": { "x": 1.5, "y": 0, "z": -3.2 },
    "obstacle": { "position": { "x": 2, "y": 2, "z": -3 } }
  }
  ```
* If `velocities` is shorter/longer than the current number of obstacles, it is normalized automatically (extra velocities ignored, missing filled with random values).
* Obstacles move within the defined `bounds` box. Adjust bounds to keep motion inside the 100×100 floor.


## 8. Reset Counter (and Simulator)

* **Endpoint:** `/reset`

* **Method:** `POST`

* **Effect:**

  * Resets server’s `collision_count` to 0.
  * Broadcasts a WebSocket `{ "command": "reset" }` to connected simulators.
  * Each simulator:

    * stops the robot,
    * resets the robot pose to origin and color to default,
    * zeros the on-screen collision counter,
    * replies with a confirmation message.

* **Response (examples):**

  * If at least one simulator is connected:

    ```json
    { "status": "reset broadcast", "collisions": 0 }
    ```
  * If no simulators are connected (server counter still reset):

    ```json
    { "status": "reset done (no simulators connected)", "collisions": 0 }
    ```

**Curl Example**

```bash
curl -X POST http://localhost:5000/reset
```

---

### Simulator Responses (added)

* **Reset Confirmation** (from simulator → server via WebSocket):

  ```json
  { "type": "confirmation", "message": "Simulator reset" }
  ```

---

### UI Notes

* The HUD in the simulator shows:

  ```
  Collisions: <number>
  ```
* This HUD is local to each simulator page, while the server’s `/collisions` count is global to the server process.

---

### Troubleshooting

* **Count not increasing?**

  * Make sure you’re using the updated **index.html** (with the HUD and WebSocket collision send) and **server.py** (with collision counting in `ws_handler`).
  * Ensure the simulator is actually colliding (obstacles present and reachable).
* **`/reset` returns “no simulators connected”?**

  * Open `index.html` in a browser first so it connects to the WebSocket server.
* **Multiple tabs open?**

  * Each tab has its own HUD counter; the server count aggregates collisions from all connected tabs.


#Additional Information

## Collision Detection
The Three.js simulator continuously checks for collisions between the robot and obstacles. When a collision is detected, the simulator stops the robot and sends a collision message via WebSocket.

## Flask & WebSocket Integration
The Flask API (running on port 5000) communicates with the WebSocket server (running on port 8080) via shared global state and the asyncio event loop.

## Troubleshooting
Port Availability: Ensure that ports 5000 and 8080 are free.

Browser Console: Check for errors if the simulator does not load or connect.

Firewall/Antivirus: Verify that your firewall or antivirus software is not blocking the connections.
