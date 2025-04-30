from ultralytics import YOLO
import cv2
import torch
import time
from PIL import Image
from IPython.display import display, Image as IPythonImage
import numpy as np
import os

# Check for GPU availability and set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable CUDA optimizations if available
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Load the model with optimizations
model = YOLO('drone_detection/train/weights/best.pt')
#drone_detection/train/weights/best.pt
model.to(device)  # Move model to GPU if available

# # Use half precision if on GPU for better performance
# if device.type != 'cpu':
#     model = model.half()  # FP16 (half precision)
#     print("Using half precision for faster inference")

# Performance optimization settings
imgsz = 640  # Lower resolution for faster inference
conf_threshold = 0.40  # Confidence threshold for detections


def process_image(image_path):
    """Process a single image and display results"""
    start_time = time.time()
    
    # Run prediction with optimized settings
    results = model(image_path, conf=conf_threshold, imgsz=imgsz)
    
    # Calculate and print inference time
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")
    
    # Extract results and process each detection
    for r in results:
        # Get the original image for annotation
        im_array = r.plot()  # get the plotted image with detections
        
        # Process each detected box
        boxes = r.boxes
        for i, box in enumerate(boxes):
            # Get the box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Calculate midpoint coordinates
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Print midpoint coordinates
            print(f"Detection {i+1}: Midpoint coordinates ({mid_x:.1f}, {mid_y:.1f})")
            
            # Optionally: Draw midpoint on the image
            cv2.circle(im_array, (int(mid_x), int(mid_y)), 5, (0, 255, 0), -1)  # Green circle
            cv2.putText(im_array, f"({mid_x:.1f}, {mid_y:.1f})", 
                       (int(mid_x) + 10, int(mid_y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the annotated image
        im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB
        im.save('results.jpg')
    
    return IPythonImage('results.jpg')


def process_video(video_path, output_path=None, max_frames=None, frame_skip=1, show_display=True):
    cap = cv2.VideoCapture(int(video_path)) if video_path.isdigit() else cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is None:
        max_frames = total_frames if total_frames > 0 else 300

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    processed_frames = 0
    start_time = time.time()
    processing_times = []

    try:
        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue  # Skip this frame

            frame_start_time = time.time()

            # Run detection
            results = model(frame, conf=conf_threshold, imgsz=imgsz)
            annotated_frame = results[0].plot()
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                cv2.circle(annotated_frame, (int(mid_x), int(mid_y)), 5, (0, 255, 0), -1)
                cv2.putText(annotated_frame, f"({mid_x:.1f}, {mid_y:.1f})", 
                            (int(mid_x) + 10, int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # FPS calculation
            frame_time = time.time() - frame_start_time
            processing_times.append(frame_time)
            if len(processing_times) > 30:
                processing_times.pop(0)
            avg_fps = 1.0 / (sum(processing_times) / len(processing_times))

            # Display
            if show_display:
                cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Video Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if output_path:
                out.write(annotated_frame)

            # GPU memory cleanup
            if device.type == 'cuda' and processed_frames % 30 == 0:
                torch.cuda.empty_cache()

            processed_frames += 1

    finally:
        cap.release()
        if output_path:
            out.release()
        if show_display:
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print(f"Processed {processed_frames} frames in {elapsed:.2f} seconds | Avg FPS: {processed_frames / elapsed:.2f}")


def process_webcam_realtime(camera_id=0, display_window=True, record_output=None, frame_skip=1):
    """
    Process webcam feed in real-time for drone detection with optimized GPU usage
    
    Args:
        camera_id: ID of the webcam (usually 0 for built-in webcam)
        display_window: Whether to show detection results in a window
        record_output: Path to save video output (None for no recording)
        frame_skip: Process every n-th frame to improve performance (1 = process all frames)
    """
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with ID {camera_id}")
        return
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Webcam: {width}x{height} at {fps}fps")
    
    # Set up video writer if recording output
    if record_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(record_output, fourcc, fps, (width, height))
        print(f"Recording to: {record_output}")
    
    # Variables for FPS calculation
    frame_count = 0
    skip_count = 0
    processing_times = []
    start_time = time.time()
    
    # Create window if display enabled
    if display_window:
        cv2.namedWindow('Drone Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Drone Detection', width, height)
    
    print("Starting real-time drone detection...")
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame from webcam")
                break
            
            frame_count += 2
            
            # Skip frames if needed for performance
            if frame_count % frame_skip != 0:
                skip_count += 2
                # Still display original frame if window enabled
                if display_window:
                    cv2.imshow('Drone Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
            frame_start = time.time()
            
            # Process frame with model
            results = model(frame, conf=conf_threshold, imgsz=imgsz)
            
            # Get annotated frame with detections
            annotated_frame = results[0].plot()
            
            # Process each detected box and display midpoints
            boxes = results[0].boxes
            print(f"Frame {frame_count}:")
            for i, box in enumerate(boxes):
                # Get the box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate midpoint coordinates
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # Print midpoint coordinates for this detection
                print(f"  Detection {i+1}: Midpoint coordinates ({mid_x:.1f}, {mid_y:.1f})")
                
                # Draw midpoint on the frame
                cv2.circle(annotated_frame, (int(mid_x), int(mid_y)), 5, (0, 255, 0), -1)  # Green circle
                cv2.putText(annotated_frame, f"({mid_x:.1f}, {mid_y:.1f})", 
                           (int(mid_x) + 10, int(mid_y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate processing time
            frame_time = time.time() - frame_start
            processing_times.append(frame_time)
            
            # Calculate and display FPS
            if len(processing_times) > 30:
                processing_times.pop(0)  # Keep only recent frames for FPS calculation
            
            avg_fps = 1.0 / (sum(processing_times) / len(processing_times))
            fps_text = f"FPS: {avg_fps:.1f}"
            
            # Add FPS counter to the frame
            cv2.putText(annotated_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Show frame if display enabled
            if display_window:
                cv2.imshow('Drone Detection', annotated_frame)
            
            # Save to video if recording
            if record_output:
                out.write(annotated_frame)
            
            # Periodically clear GPU cache
            if device.type == 'cuda' and frame_count % 30 == 0:
                torch.cuda.empty_cache()
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame on 's' key press
                save_path = f"detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"Saved frame to {save_path}")
                
    except KeyboardInterrupt:
        print("\nWebcam detection interrupted by user")
        
    finally:
        # Clean up
        cap.release()
        if record_output and 'out' in locals():
            out.release()
        if display_window:
            cv2.destroyAllWindows()
        
        # Show statistics
        elapsed_time = time.time() - start_time
        if processing_times:
            final_fps = 1.0 / (sum(processing_times) / len(processing_times))
            print(f"\nAverage FPS: {final_fps:.2f}")
        
        print(f"Processed {frame_count - skip_count} frames in {elapsed_time:.2f} seconds")
        print(f"Skipped {skip_count} frames")
                
        
def run_detection(mode='video', source_path=None, output_path=None, max_frames=None, camera_id=0, frame_skip=1):
    """Run detection in specified mode with given paths"""
    if mode.lower() == 'image':
        if source_path:
            display(process_image(source_path))
        else:
            print("Please provide an image path")
    
    elif mode.lower() == 'video':
        if source_path:
            process_video(source_path, output_path, max_frames)
        else:
            print("Please provide a video path or '0' for webcam")
    
    elif mode.lower() == 'webcam':
        # Use the new dedicated webcam mode
        process_webcam_realtime(
            camera_id=camera_id, 
            display_window=True, 
            record_output=output_path,
            frame_skip=frame_skip
        )
    
    else:
        print("Invalid mode. Use 'image', 'video', or 'webcam'")

if __name__ == "__main__":
    # Example usage:
    # For webcam detection:
    # run_detection(mode='webcam', camera_id=0, output_path='webcam_recording.mp4', frame_skip=2)
    
    # For image processing:
    # run_detection(mode='image', source_path='test/drone_test_1.jpg')
    
    # For video processing:
    # run_detection(mode='video', source_path='test/drone_test.mp4', output_path='output_video.mp4')
    
    # Default to video mode with provided paths
    run_detection(mode='video',source_path="test_samples/drone_test4.mp4" ,output_path='output_video/output_new_video4.mp4')