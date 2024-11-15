from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import logging
import time

class YOLOInference:
    """A class to handle YOLO model inference with additional features and error handling"""
    
    def __init__(self, model_path, conf_threshold=0.25, save_results=False, output_dir='outputs'):
        """
        Initialize the YOLO inference handler
        
        Args:
            model_path (str): Path to the YOLO model .pt file
            conf_threshold (float): Confidence threshold for detections (0-1)
            save_results (bool): Whether to save annotated images
            output_dir (str): Directory to save results if save_results is True
        """
        self.conf_threshold = conf_threshold
        self.save_results = save_results
        self.output_dir = Path(output_dir)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model
        try:
            self.model = YOLO(model_path)
            self.logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
            
        # Create output directory if saving results
        if self.save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_image(self, image_path, display_results=True):
        """
        Process a single image with the YOLO model
        
        Args:
            image_path (str): Path to the input image
            display_results (bool): Whether to display results in a window
            
        Returns:
            dict: Dictionary containing detection results and performance metrics
        """
        try:
            # Verify image exists
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Time the inference
            start_time = time.time()
            
            # Run inference
            results = self.model(image_path, conf=self.conf_threshold)
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            # Process and log results
            detections = []
            for result in results:
                boxes = result.boxes
                
                self.logger.info(f"\nProcessed {Path(image_path).name}")
                self.logger.info(f"Found {len(boxes)} detections")
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    # Get class and confidence
                    conf = float(box.conf)
                    cls = int(box.cls)
                    class_name = self.model.names[cls]
                    
                    detection = {
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    }
                    detections.append(detection)
                    
                    self.logger.info(
                        f"Class: {class_name}, Confidence: {conf:.2f}, "
                        f"Bbox: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})"
                    )
            
            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Save results if requested
            if self.save_results:
                output_path = self.output_dir / f"detected_{Path(image_path).name}"
                cv2.imwrite(str(output_path), annotated_frame)
                self.logger.info(f"Saved annotated image to {output_path}")
            
            # Display results if requested
            if display_results:
                cv2.imshow("YOLOv8 Inference", annotated_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Prepare return dictionary
            return {
                'detections': detections,
                'inference_time': inference_time,
                'num_detections': len(detections),
                'annotated_frame': annotated_frame
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise

def main():
    """Example usage of the YOLOInference class"""
    # Define paths
    model_path = r'C:\cornerstonex\machine_learning\Mobility-Aids-Detection\runs\detect\train\weights\best.pt'
    image_path = r'C:\cornerstonex\machine_learning\Mobility-Aids-Detection\dataset\images\test\02_jpeg.rf.0352c7dbff7a331dd8ea3f828c8ef62e.jpg'
    
    try:
        # Initialize inference handler
        yolo = YOLOInference(
            model_path=model_path,
            conf_threshold=0.25,
            save_results=True,
            output_dir='yolo_outputs'
        )
        
        # Process image
        results = yolo.process_image(image_path)
        
        # Print performance metrics
        print(f"\nPerformance Metrics:")
        print(f"Inference time: {results['inference_time']:.3f} seconds")
        print(f"Number of detections: {results['num_detections']}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()