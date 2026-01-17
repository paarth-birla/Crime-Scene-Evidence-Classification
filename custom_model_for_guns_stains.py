from ultralytics import YOLO


def train_custom_model():
    # 1. Load a pre-trained model to start from
    # We use 'yolov8m.pt' (Medium) as the base.
    # It already knows shapes/edges, which speeds up learning guns/blood.
    model = YOLO('yolov8m.pt')

    # 2. Train the model
    # data: Path to your data.yaml file (defined in Step 1)
    # epochs: 50-100 is usually good for a specialized task
    # imgsz: 640 is standard
    print("[INFO] Starting training for Guns and Blood Stains...")

    results = model.train(
        data='datasets/gun_blood_data/data.yaml',
        epochs=10,
        imgsz=416,
        cache=True,
        workers=4,
        freeze=10,
        batch=16,
        name='gun_blood_model',  # Results saved to runs/detect/gun_blood_model
        patience=10  # Stop early if no improvement
    )

    print("[SUCCESS] Training complete.")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")


if __name__ == '__main__':
    # Fix for Windows multiprocessing usage
    train_custom_model()