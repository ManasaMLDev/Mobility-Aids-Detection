from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import logging
import time
from typing import Union, Optional
from tqdm import tqdm

class YOLOInference:
    """A class to handle YOLO model inference for video files"""
    
    def __init__(
        self, 
        model_path: str, 
        conf_threshold: float = 0.25,
        output_dir: str = 'outputs',
        device: str = None  # 'cpu' or 'cuda' or None for auto
    ):
        """
        Initialize the YOLO inference handler
        
        Args:
            model_path (str): Path to the YOLO model .pt file
            conf_threshold (float): Confidence threshold for detections (0-1)
            output_dir (str): Directory to save results
            device (str): Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.conf_threshold = conf_threshold
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        try:
            self.model = YOLO(model_path)
            if device:
                self.model.to(device)
            self.logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def process_video(
        self,
        video_path: str,
        output_filename: Optional[str] = None,
        display_preview: bool = True
    ) -> dict:
        """
        Process a video file with the YOLO model
        
        Args:
            video_path (str): Path to input video file
            output_filename (str): Filename for processed video (if None, auto-generated)
            display_preview (bool): Whether to display preview window
            
        Returns:
            dict: Processing statistics
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        try:
            # Open video capture
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Generate output filename if not provided
            if output_filename is None:
                output_filename = f"{video_path.stem}_processed.mp4"
            
            # Setup video writer
            output_path = self.output_dir / output_filename
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height)
            )
            
            # Initialize statistics
            stats = {
                'total_frames': total_frames,
                'processed_frames': 0,
                'total_detections': 0,
                'processing_times': [],
                'video_info': {
                    'original_path': str(video_path),
                    'output_path': str(output_path),
                    'resolution': f"{frame_width}x{frame_height}",
                    'fps': fps,
                    'duration': total_frames / fps
                }
            }
            
            self.logger.info(f"\nProcessing video: {video_path.name}")
            self.logger.info(f"Resolution: {frame_width}x{frame_height}")
            self.logger.info(f"FPS: {fps}")
            self.logger.info(f"Total frames: {total_frames}")
            self.logger.info(f"Output path: {output_path}")
            
            # Process frames with progress bar
            pbar = tqdm(total=total_frames, desc="Processing frames")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                start_time = time.time()
                
                # Run inference
                results = self.model(frame, conf=self.conf_threshold)
                
                # Get detections
                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        
                        # Get class and confidence
                        conf = float(box.conf)
                        cls = int(box.cls)
                        class_name = self.model.names[cls]
                        
                        detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': (int(x1), int(y1), int(x2), int(y2))
                        })
                
                # Get annotated frame
                annotated_frame = results[0].plot()
                
                # Add processing information to frame
                processing_time = time.time() - start_time
                fps_current = 1.0 / processing_time if processing_time > 0 else 0
                
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps_current:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Write frame
                writer.write(annotated_frame)
                
                # Display preview if requested
                if display_preview:
                    # Resize preview for better viewing
                    scale = min(1.0, 1024/frame_width)  # Limit preview width to 1024
                    if scale < 1.0:
                        preview_frame = cv2.resize(
                            annotated_frame, 
                            None, 
                            fx=scale, 
                            fy=scale
                        )
                    else:
                        preview_frame = annotated_frame
                        
                    cv2.imshow("Processing Preview", preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Processing cancelled by user")
                        break
                
                # Update statistics
                stats['processed_frames'] += 1
                stats['total_detections'] += len(detections)
                stats['processing_times'].append(processing_time)
                
                # Update progress bar
                pbar.update(1)
            
            # Calculate final statistics
            processing_times = np.array(stats['processing_times'])
            stats['performance'] = {
                'average_fps': 1.0 / processing_times.mean(),
                'max_fps': 1.0 / processing_times.min(),
                'min_fps': 1.0 / processing_times.max(),
                'average_detections_per_frame': stats['total_detections'] / stats['processed_frames'],
                'total_processing_time': processing_times.sum()
            }
            
            # Clean up
            pbar.close()
            cap.release()
            writer.release()
            if display_preview:
                cv2.destroyAllWindows()
            
            # Log final statistics
            self.logger.info("\nProcessing completed!")
            self.logger.info(f"Processed {stats['processed_frames']} frames")
            self.logger.info(f"Average FPS: {stats['performance']['average_fps']:.1f}")
            self.logger.info(f"Total detections: {stats['total_detections']}")
            self.logger.info(f"Output saved to: {output_path}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            raise
        finally:
            # Ensure everything is properly closed
            try:
                pbar.close()
                cap.release()
                writer.release()
                cv2.destroyAllWindows()
            except:
                pass

def main():
    """Example usage of the YOLOInference class for video processing"""
    # Define paths
    model_path = r'C:\cornerstonex\machine_learning\Mobility-Aids-Detection\runs\detect\train\weights\best.pt'
    video_path = r'C:\cornerstonex\machine_learning\Mobility-Aids-Detection\test\cane1.mp4'  # Replace with your video path
    
    try:
        # Initialize inference handler
        yolo = YOLOInference(
            model_path=model_path,
            conf_threshold=0.60,
            output_dir='yolo_outputs'
        )
        
        # Process video file
        stats = yolo.process_video(
            video_path=video_path,
            output_filename='processed_video.mp4',
            display_preview=True
        )
        
        # Print detailed statistics
        print("\nProcessing Statistics:")
        print(f"Video Resolution: {stats['video_info']['resolution']}")
        print(f"Original FPS: {stats['video_info']['fps']}")
        print(f"Duration: {stats['video_info']['duration']:.1f} seconds")
        print(f"\nPerformance:")
        print(f"Average FPS: {stats['performance']['average_fps']:.1f}")
        print(f"Peak FPS: {stats['performance']['max_fps']:.1f}")
        print(f"Total processing time: {stats['performance']['total_processing_time']:.1f} seconds")
        print(f"\nDetections:")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Average detections per frame: {stats['performance']['average_detections_per_frame']:.2f}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()