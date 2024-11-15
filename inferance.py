from ultralytics import YOLO
import cv2
import os

def run_inference(model_path, input_path, conf_threshold=0.25):
    """
    Run YOLO inference on an image or video and visualize results.
    
    Args:
        model_path (str): Path to the YOLO model .pt file.
        input_path (str): Path to the input image or video file.
        conf_threshold (float): Confidence threshold for detections.
    """
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Determine if the input is an image or video
    is_image = input_path.lower().endswith(('.jpg', '.jpeg', '.png'))
    
    # Helper function to process results
    def process_results(results):
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Box coordinates
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = model.names[cls]
                print(f"Class: {class_name}, Confidence: {conf:.2f}")
                print(f"Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")
        return results[0].plot()  # Return annotated frame

    # Handle image input
    if is_image:
        results = model(input_path, conf=conf_threshold)
        annotated_frame = process_results(results)
        cv2.imshow("YOLOv11 Inference - Image", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Handle video input
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference and process each frame
            results = model(frame, conf=conf_threshold)
            annotated_frame = process_results(results)
            
            cv2.imshow("YOLOv11 Inference - Video", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    model_path = r'C:\Users\VijayRajput\Downloads\Disabled_640S_Best.pt'
    input_path = r'DisabledDetect\test\images\02_jpeg.rf.0352c7dbff7a331dd8ea3f828c8ef62e.jpg'  # Change to a video path for video input
    run_inference(model_path, input_path)
