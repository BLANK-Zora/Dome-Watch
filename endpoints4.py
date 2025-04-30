from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import time
import os
import logging
from typing import Generator
from fastapi.middleware.cors import CORSMiddleware
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta

import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("yolo-detection")

app = FastAPI(title="YOLO Detection API", 
              description="API for object detection using YOLOv8",
              version="1.0.0")

# Production-ready CORS settings - adjust allowed_origins for your environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Restrict to your frontend's origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Restrict to needed methods
    allow_headers=["*"],
)

# Create directories
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize model - lazy loading to avoid startup delay
model = None
device = None
CONF_THRESHOLD = 0.30
IMG_SIZE = 416
MODEL_PATH = 'drone_detection/train/weights/best.pt'

# Stream control variables
stream_active = False
stream_lock = threading.Lock()



def get_model():
    """Lazy loading of the YOLO model"""
    global model, device
    if model is None:
        try:
            logger.info("Loading YOLO model...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = YOLO(MODEL_PATH).to(device)
            logger.info(f"Model loaded successfully using {device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    return model

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Create the index.html template
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Detection Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .video-container {
            margin: 20px 0;
            text-align: center;
        }
        .webcam-feed {
            max-width: 100%;
            border: 2px solid #ddd;
            border-radius: 4px;
        }
        .options {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
        }
        .webcam-controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }
        .start-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .stop-btn {
            background-color: #f44336;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .start-btn:hover {
            background-color: #45a049;
        }
        .stop-btn:hover {
            background-color: #d32f2f;
        }
        .upload-form {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        input[type="file"] {
            margin: 10px 0;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .result-container {
            margin-top: 20px;
            text-align: center;
        }
        .result-image {
            max-width: 100%;
            border: 2px solid #ddd;
            border-radius: 4px;
        }
        .status-indicator {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            color: white;
            font-weight: bold;
            margin-left: 10px;
        }
        .status-active {
            background-color: #4CAF50;
        }
        .status-inactive {
            background-color: #f44336;
        }
        #stream-placeholder {
            width: 640px;
            height: 480px;
            background-color: #eee;
            border: 2px solid #ddd;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>YOLO Detection Dashboard</h1>
        
        <div class="options">
            <button onclick="showWebcam()">Webcam Detection</button>
            <button onclick="showImageUpload()">Image Upload</button>
            <button onclick="showVideoUpload()">Video Upload</button>
        </div>
        
        <div id="webcam-section" class="video-container">
            <h2>Live Webcam Detection <span id="stream-status" class="status-indicator status-inactive">Inactive</span></h2>
            <div id="stream-container">
                <div id="stream-placeholder">Stream not active. Click Start to begin.</div>
            </div>
            <div class="webcam-controls">
                <button id="start-stream" class="start-btn" onclick="startWebcamStream()">Start Stream</button>
                <button id="stop-stream" class="stop-btn" onclick="stopWebcamStream()" disabled>Stop Stream</button>
            </div>
        </div>
        
        <div id="image-upload-section" class="upload-form" style="display:none;">
            <h2>Upload Image for Detection</h2>
            <form id="image-form" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Detect Objects</button>
            </form>
            <div id="image-result" class="result-container"></div>
        </div>
        
        <div id="video-upload-section" class="upload-form" style="display:none;">
            <h2>Upload Video for Detection</h2>
            <form id="video-form" enctype="multipart/form-data">
                <input type="file" name="file" accept="video/*" required>
                <button type="submit">Process Video</button>
            </form>
            <div id="video-result" class="result-container"></div>
        </div>
    </div>

    <script>
        // Global variables for stream management
        let streamActive = false;
        let streamImage = null;
        
        function showWebcam() {
            document.getElementById('webcam-section').style.display = 'block';
            document.getElementById('image-upload-section').style.display = 'none';
            document.getElementById('video-upload-section').style.display = 'none';
        }
        
        function showImageUpload() {
            document.getElementById('webcam-section').style.display = 'none';
            document.getElementById('image-upload-section').style.display = 'block';
            document.getElementById('video-upload-section').style.display = 'none';
        }
        
        function showVideoUpload() {
            document.getElementById('webcam-section').style.display = 'none';
            document.getElementById('image-upload-section').style.display = 'none';
            document.getElementById('video-upload-section').style.display = 'block';
        }
        
        async function startWebcamStream() {
            try {
                // Update status before starting stream
                document.getElementById('stream-status').className = 'status-indicator status-active';
                document.getElementById('stream-status').textContent = 'Active';
                
                // Enable stop button, disable start button
                document.getElementById('start-stream').disabled = true;
                document.getElementById('stop-stream').disabled = false;
                
                // Start the stream on the server
                await fetch('/stream/start');
                
                // Create image element if it doesn't exist
                if (!streamImage) {
                    streamImage = document.createElement('img');
                    streamImage.className = 'webcam-feed';
                    streamImage.alt = 'Webcam Feed';
                }
                
                // Set the source with cache-busting parameter
                streamImage.src = '/stream/webcam?t=' + new Date().getTime();
                
                // Replace placeholder with stream
                const streamContainer = document.getElementById('stream-container');
                streamContainer.innerHTML = '';
                streamContainer.appendChild(streamImage);
                
                streamActive = true;
            } catch (error) {
                console.error('Error starting stream:', error);
                alert('Error starting webcam stream');
                stopWebcamStream();
            }
        }
        
        async function stopWebcamStream() {
            try {
                // Stop the stream on the server
                await fetch('/stream/stop');
                
                // Update UI
                document.getElementById('stream-status').className = 'status-indicator status-inactive';
                document.getElementById('stream-status').textContent = 'Inactive';
                
                // Enable start button, disable stop button
                document.getElementById('start-stream').disabled = false;
                document.getElementById('stop-stream').disabled = true;
                
                // Replace stream with placeholder
                const streamContainer = document.getElementById('stream-container');
                streamContainer.innerHTML = '<div id="stream-placeholder">Stream not active. Click Start to begin.</div>';
                
                streamActive = false;
            } catch (error) {
                console.error('Error stopping stream:', error);
            }
        }
        
        document.getElementById('image-form').addEventListener('submit', async (event) => {
            event.preventDefault();
            const formData = new FormData(event.target);
            
            try {
                const response = await fetch('/predict/image', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                const resultDiv = document.getElementById('image-result');
                resultDiv.innerHTML = `
                    <h3>Detection Results</h3>
                    <img src="${data.result_image_url}?t=${new Date().getTime()}" class="result-image" alt="Detection Result">
                    <p>Detected ${data.midpoints.length} objects</p>
                `;
            } catch (error) {
                console.error('Error:', error);
                alert('Error processing image');
            }
        });
        
        document.getElementById('video-form').addEventListener('submit', async (event) => {
            event.preventDefault();
            const formData = new FormData(event.target);
            const resultDiv = document.getElementById('video-result');
            
            resultDiv.innerHTML = '<p>Processing video... This may take a while.</p>';
            
            try {
                const response = await fetch('/predict/video', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const videoBlob = await response.blob();
                    const videoUrl = URL.createObjectURL(videoBlob);
                    
                    resultDiv.innerHTML = `
                        <h3>Processed Video</h3>
                        <video controls width="100%" src="${videoUrl}"></video>
                        <a href="${videoUrl}" download="detection_result.mp4">Download Video</a>
                    `;
                } else {
                    resultDiv.innerHTML = '<p>Error processing video.</p>';
                }
            } catch (error) {
                console.error('Error:', error);
                resultDiv.innerHTML = '<p>Error processing video.</p>';
            }
        });
        
        // Clean up when page is closed or refreshed
        window.addEventListener('beforeunload', async () => {
            if (streamActive) {
                await fetch('/stream/stop');
            }
        });
    </script>
</body>
</html>
    """)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting application...")
    
    # Create cleanup task for temporary files
    @app.on_event("startup")
    async def cleanup_temp_files():
        logger.info("Cleaning up temporary files...")
        try:
            # Delete files older than 1 hour
            cleanup_old_files(TEMP_DIR, hours=1)
        except Exception as e:
            logger.error(f"Error cleaning temporary files: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application...")
    # Release CUDA memory if used
    if device and device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

def cleanup_old_files(directory: str, hours: int = 1):
    """Delete files older than the specified hours"""
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_modified_time < cutoff_time:
                try:
                    os.remove(file_path)
                    logger.debug(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")

@contextmanager
def video_capture(source):
    """Context manager for video capture to ensure resources are released"""
    cap = None
    try:
        # Handle numeric string as integer for webcam sources
        if isinstance(source, str) and source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        
        yield cap
    finally:
        if cap is not None:
            cap.release()

@contextmanager
def video_writer(output_path, fps, width, height):
    """Context manager for video writer to ensure resources are released"""
    writer = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        yield writer
    finally:
        if writer is not None:
            writer.release()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the homepage with webcam stream"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/stream/start")
async def start_stream():
    """Start the webcam stream"""
    global stream_active
    with stream_lock:
        stream_active = True
    return {"status": "started"}

@app.get("/stream/stop")
async def stop_stream():
    """Stop the webcam stream"""
    global stream_active
    with stream_lock:
        stream_active = False
    return {"status": "stopped"}

@app.get("/stream/status")
async def stream_status():
    """Get the current stream status"""
    global stream_active
    with stream_lock:
        is_active = stream_active
    return {"active": is_active}

def predict_image(contents: bytes) -> tuple:
    """Process an image and return detection results"""
    try:
        # Convert bytes to image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Get model and run prediction
        model = get_model()
        results = model(img, conf=CONF_THRESHOLD, imgsz=IMG_SIZE)
        
        # Create annotated image with midpoints
        annotated = results[0].plot()
        
        # Add midpoints to the image
        midpoints = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            midpoints.append({"mid_x": round(mid_x, 1), "mid_y": round(mid_y, 1)})
            
            # Draw midpoint on the image
            cv2.circle(annotated, (int(mid_x), int(mid_y)), 5, (0, 255, 0), -1)
            cv2.putText(annotated, f"({mid_x:.1f}, {mid_y:.1f})", 
                        (int(mid_x)+10, int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Generate unique filename
        result_path = os.path.join(TEMP_DIR, f"result_{uuid.uuid4()}.jpg")
        cv2.imwrite(result_path, annotated)
        
        return result_path, midpoints
    except Exception as e:
        logger.error(f"Error in predict_image: {e}")
        raise RuntimeError(f"Image processing failed: {str(e)}")

@app.post("/predict/image")
async def predict_image_api(file: UploadFile = File(...)):
    """API endpoint for image prediction"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        result_path, midpoints = predict_image(contents)
        
        # Get the filename only for the URL
        result_filename = os.path.basename(result_path)
        
        return JSONResponse({
            "result_image_url": f"/result/image/{result_filename}",
            "midpoints": midpoints
        })
    except Exception as e:
        logger.error(f"Error in predict_image_api: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result/image/{filename}")
def get_image_result(filename: str):
    """Serve the result image"""
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result image not found")
    return FileResponse(file_path, media_type="image/jpeg")

@app.post("/predict/video")
async def predict_video_api(file: UploadFile = File(...)):
    """API endpoint for video prediction"""
    try:
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Create unique filenames
        input_filename = f"input_{uuid.uuid4()}.mp4"
        output_filename = f"output_{uuid.uuid4()}.mp4"
        input_path = os.path.join(TEMP_DIR, input_filename)
        output_path = os.path.join(TEMP_DIR, output_filename)
        
        # Save uploaded video
        contents = await file.read()
        with open(input_path, "wb") as f:
            f.write(contents)
        
        # Process video
        try:
            process_video(input_path, output_path=output_path, show_display=False)
            
            # Check if output file exists and has content
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Video processing failed to create output")
                
            return FileResponse(output_path, media_type="video/mp4")
        finally:
            # Clean up input file
            if os.path.exists(input_path):
                os.remove(input_path)
    except Exception as e:
        logger.error(f"Error in predict_video_api: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def webcam_stream() -> Generator[bytes, None, None]:
    """Generate frames from webcam with object detection"""
    global stream_active
    
    try:
        model = get_model()
        error_frame = None
        
        with video_capture(0) as cap:
            # Create an error frame in case we need it
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Stream stopped", (180, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Check if stream is active before starting
            is_active = False
            with stream_lock:
                is_active = stream_active
                
            if not is_active:
                # Return an inactive frame if stream is not active
                _, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                return
            
            # Process frames while stream is active
            while True:
                # Check if stream should continue
                with stream_lock:
                    if not stream_active:
                        break
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to get frame from webcam")
                    break
                
                try:
                    # Run detection
                    results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE)
                    annotated = results[0].plot()
                    
                    # Draw midpoints
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        mid_x = (x1 + x2) / 2
                        mid_y = (y1 + y2) / 2
                        cv2.circle(annotated, (int(mid_x), int(mid_y)), 5, (0, 255, 0), -1)
                        cv2.putText(annotated, f"({mid_x:.1f}, {mid_y:.1f})", 
                                    (int(mid_x)+10, int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Encode frame
                    _, buffer = cv2.imencode('.jpg', annotated)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                except Exception as frame_error:
                    logger.error(f"Error processing frame: {frame_error}")
                    # Return an error frame
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "Error processing frame", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    _, buffer = cv2.imencode('.jpg', error_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                # Periodic CUDA cleanup
                if device and device.type == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Send a final frame indicating stream is stopped
            _, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
    except Exception as e:
        logger.error(f"Error in webcam_stream: {e}")
        # Create an error frame if the original one wasn't created
        if error_frame is None:
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        cv2.putText(error_frame, f"Stream error: {str(e)}", (50, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        # Make sure stream is marked as inactive when exiting
        with stream_lock:
            stream_active = False

@app.get("/stream/webcam")
def stream_webcam():
    """API endpoint for webcam streaming"""
    try:
        return StreamingResponse(webcam_stream(), media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        logger.error(f"Error in stream_webcam: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def process_video(video_path, output_path=None, max_frames=None, frame_skip=1, show_display=True):  # Set default to True
    """Process a video file with object detection and display it in real-time"""
    try:
        model = get_model()
        
        with video_capture(video_path) as cap:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frames = max_frames if max_frames else (total_frames if total_frames > 0 else 300)
            
            # Create display window
            if show_display:
                cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Detection", width, height)
            
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            processed_frames = 0
            processing_times = []
            
            while processed_frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue
                
                frame_start = time.time()
                results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE)
                annotated = results[0].plot()
                
                # Draw midpoints
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    cv2.circle(annotated, (int(mid_x), int(mid_y)), 5, (0, 255, 0), -1)
                    cv2.putText(annotated, f"({mid_x:.1f}, {mid_y:.1f})", 
                                (int(mid_x)+10, int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                processing_times.append(frame_time)
                if len(processing_times) > 30:
                    processing_times.pop(0)
                avg_fps = 1.0 / (sum(processing_times) / len(processing_times))
                
                # Add FPS counter
                cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Always display the processed frame
                if show_display:
                    cv2.imshow("Detection", annotated)
                    # Add a delay to make it visible (can adjust based on original video fps)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Write to output if specified
                if writer is not None:
                    writer.write(annotated)
                
                # CUDA cleanup
                if device and device.type == 'cuda' and torch.cuda.is_available() and processed_frames % 30 == 0:
                    torch.cuda.empty_cache()
                
                processed_frames += 1
            
            # Clean up
            if writer is not None:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()
                
    except Exception as e:
        logger.error(f"Error in process_video: {e}")
        raise RuntimeError(f"Video processing failed: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    # Use production server with workers
    uvicorn.run("endpoints4:app", host="0.0.0.0", port=8000, reload=False, workers=1)