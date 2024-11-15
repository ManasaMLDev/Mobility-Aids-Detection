from ultralytics import YOLO
import os
from pathlib import Path
import yaml

def create_dataset_config(
    train_path,
    val_path,
    class_names,
    output_path='dataset.yaml'
):
    """
    Create YAML configuration file for training
    """
    data = {
        'path': os.path.dirname(train_path),  # dataset root dir
        'train': train_path,  # train images
        'val': val_path,     # val images
        'names': class_names  # class names
    }
    
    # Save YAML file
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    return output_path

def train_yolo(
    data_yaml,
    model_type='yolov8n.pt',  # or 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
    epochs=100,
    imgsz=640,
    batch_size=16,
    device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    project='runs/train',
    name='exp',
    resume=False,
    pretrained=True
):
    """
    Train YOLO model with specified parameters
    """
    # Initialize model
    if pretrained:
        model = YOLO(model_type)
    else:
        # Start with new model
        model = YOLO(model_type).load('empty')
    
    # Training arguments
    args = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        resume=resume,
        verbose=True,
        save=True,
        plots=True,  # Save training plots
        save_period=10,  # Save checkpoint every X epochs
        patience=50,  # Early stopping patience
        seed=42,
    )
    
    # Train the model
    results = model.train(**args)
    
    return model, results

def validate_model(model, data_yaml):
    """
    Validate the trained model
    """
    # Run validation
    metrics = model.val(data=data_yaml)
    
    return metrics

if __name__ == "__main__":
    # Example usage
    
    # 1. Define your dataset paths and classes
    train_path = "path/to/train/images" "C:\cornerstonex\machine_learning\Mobility-Aids-Detection\dataset\"
    val_path = "path/to/val/images"
    class_names = ['mobility_aid', 'wheelchair', 'crutches', 'walker']
    
    # 2. Create dataset configuration
    data_yaml = create_dataset_config(
        train_path=train_path,
        val_path=val_path,
        class_names=class_names
    )
    
    # 3. Train the model
    model, results = train_yolo(
        data_yaml=data_yaml,
        model_type='yolov8s.pt',  # Small model
        epochs=100,
        imgsz=640,
        batch_size=16,
        device='0',  # Use first GPU
        project='mobility_aids_detection',
        name='experiment1'
    )
    
    # 4. Validate the model
    metrics = validate_model(model, data_yaml)
    
    # 5. Print validation metrics
    print("\nValidation Metrics:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.p:.3f}")
    print(f"Recall: {metrics.box.r:.3f}")