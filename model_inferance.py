from ultralytics import YOLO
import cv2
import numpy as np

def run_inference(model_path, image_path, conf_threshold=0.25):
    """
    Run YOLO inference on an image and visualize results
    
    Args:
        model_path (str): Path to the YOLO model .pt file
        image_path (str): Path to the input image
        conf_threshold (float): Confidence threshold for detections
    """
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path, conf=conf_threshold)
    
    # Process results
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        
        print("\nDetections:")
        print("-----------")
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]  # box with xyxy format
            
            # Get class and confidence
            conf = float(box.conf)
            cls = int(box.cls)
            class_name = model.names[cls]
            
            print(f"Class: {class_name}, Confidence: {conf:.2f}")
            print(f"Bounding Box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
    
    # Plot results
    annotated_frame = results[0].plot()
    
    # Display results
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return results

# Example usage
if __name__ == "__main__":
    model_path = r'C:\cornerstonex\machine_learning\Mobility-Aids-Detection\models\Disabled_640S_Best.pt'
    image_path = r'C:\cornerstonex\machine_learning\Mobility-Aids-Detection\dataset\images\test\02_jpeg.rf.0352c7dbff7a331dd8ea3f828c8ef62e.jpg'
    
    results = run_inference(model_path, image_path)