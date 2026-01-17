import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import glob
import config


class EnsembleEvidenceDetector:
    def __init__(self, standard_weights='yolov8l.pt', custom_weights='best.pt'):
        """
        Initializes BOTH models to cover all evidence types.
        """
        print("[INIT] Loading Standard COCO Model (People, Knives, etc.)...")
        self.model_standard = YOLO(standard_weights)

        if os.path.exists(custom_weights):
            print(f"[INIT] Loading Custom Forensic Model ({custom_weights})...")
            self.model_custom = YOLO(custom_weights)

        else:
            print(f"[WARNING] Custom weights '{custom_weights}' not found! Gun/Blood detection will be skipped.")
            self.model_custom = None

        # --- CONFIGURATION ---
        self.VISUAL_CUTOFF = 0.30  # Only draw on image if > 35%

        # 1. Standard Model Targets (COCO IDs)
        self.std_classes = {
            # PERSON
            0: "Person",
            # BAGGAGE
            24: "Backpack", 26: "Handbag", 28: "Suitcase",
            # KITCHEN / DRUGS / POISON
            39: "Bottle", 40: "Wine Glass", 41: "Cup", 45: "Bowl",
            # WEAPONS
            43: "Knife", 76: "Scissors",
            # ELECTRONICS (Digital Forensics)
            67: "Cell Phone", 73: "Laptop", 63: "TV/Monitor", 66: "Keyboard", 64: "Mouse"
        }

        # 2. Custom Model Targets (Your Trained IDs)
        self.cust_classes = {
            0: "Gun",
            1: "Blood Stain"
        }

        # 3. Color Palette
        self.colors = {
            "biohazard": (0, 0, 139),  # Dark Red (Blood)
            "weapon_gun": (0, 0, 255),  # Bright Red (Gun)
            "weapon_knife": (0, 69, 255),  # Orange-Red (Knife)
            "person": (255, 255, 0),  # Cyan (Person)
            "digital": (255, 0, 0),  # Blue (Phone)
            "general": (0, 255, 255),  # Yellow (Bottle/Suitcase)
            "text": (255, 255, 255),  # White
            "bg_label": (50, 50, 50)  # Dark Grey
        }

    def process_directory(self, input_dir, output_root=config.OUTPUT_DIR):
        csv_dir = os.path.join(output_root, "evidence_logs")
        visuals_dir = os.path.join(output_root, "visuals")
        input_dir = os.path.join(output_root, "input")
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(visuals_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)

        image_files = glob.glob(input_dir)

        print(f"[INFO] Found {len(image_files)} images. Starting Ensemble Scan...")

        for img_path in image_files:
            self._analyze_image(img_path, csv_dir, visuals_dir)

        print(f"\n[COMPLETE] Results saved to '{output_root}/'")

    def _analyze_image(self, image_path, csv_dir=config.CSV_DIR, visuals_dir=config.VISUALS_DIR):
        image = cv2.imread(image_path)
        if image is None: return
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Get Image Dimensions for Boundary Checks
        img_h, img_w = image.shape[:2]

        master_log = []
        log_args = {"project": "yolo_internal_logs", "name": "inference", "exist_ok": True}

        # --- PASS 1: STANDARD MODEL ---
        res_std = self.model_standard.predict(image, conf=0.001, iou=0.5, verbose=False, **log_args)[0]
        for box in res_std.boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.std_classes:
                master_log.append({
                    "Source": "Standard_Model",
                    "Label": self.std_classes[cls_id],
                    "Conf": float(box.conf[0]),
                    "Box": box.xyxy[0].tolist()
                })

        # --- PASS 2: CUSTOM MODEL (Guns/Blood) ---
        if self.model_custom:
            res_cust = self.model_custom.predict(image, conf=0.001, iou=0.5, verbose=False, **log_args)[0]
            for box in res_cust.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.cust_classes:
                    master_log.append({
                        "Source": "Custom_Model",
                        "Label": self.cust_classes[cls_id],
                        "Conf": float(box.conf[0]),
                        "Box": box.xyxy[0].tolist()
                    })

        # --- PASS 3: PROCESSING & VISUALIZATION ---
        annotated_img = image.copy()
        csv_data = []

        for item in master_log:
            label = item['Label']
            conf = item['Conf']
            x1, y1, x2, y2 = map(int, item['Box'])

            # Clamp coordinates to stay within image (Just in case)
            x1 = max(0, x1);
            y1 = max(0, y1)
            x2 = min(img_w, x2);
            y2 = min(img_h, y2)

            # A. Add to CSV Data
            csv_data.append({
                "Timestamp": datetime.now().isoformat(),
                "Image": base_name,
                "Model_Source": item['Source'],
                "Evidence_Type": label,
                "Confidence_Score": conf,
                "Confidence_Text": f"{conf:.2%}",
                "Visualized": "YES" if conf > self.VISUAL_CUTOFF else "NO",
                "Coords": [x1, y1, x2, y2]
            })

            # B. Draw Visuals (High Confidence Only)
            if conf > self.VISUAL_CUTOFF:
                # Color Selection
                if "Blood" in label:
                    c = self.colors["biohazard"]
                elif "Gun" in label:
                    c = self.colors["weapon_gun"]
                elif "Knife" in label:
                    c = self.colors["weapon_knife"]
                elif "Person" in label:
                    c = self.colors["person"]
                elif "Phone" in label or "Laptop" in label:
                    c = self.colors["digital"]
                else:
                    c = self.colors["general"]

                # Draw Box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), c, 2)

                # --- BOUNDARY AWARE LABEL DRAWING ---
                lbl = f"{label} {conf:.0%}"
                # Get text size
                (text_w, text_h), baseline = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                # Default: Draw ABOVE the box
                # Coordinates for background rectangle
                bg_x1 = x1
                bg_y1 = y1 - text_h - 10
                bg_x2 = x1 + text_w + 10
                bg_y2 = y1

                # Coordinate for text baseline
                text_x = x1 + 5
                text_y = y1 - 5

                # CHECK 1: Top Boundary (If text goes off top edge)
                if y1 - text_h - 10 < 0:
                    # FLIP: Draw INSIDE/BELOW the top line
                    bg_y1 = y1
                    bg_y2 = y1 + text_h + 10
                    text_y = y1 + text_h + 5

                # CHECK 2: Right Boundary (If text goes off right edge)
                if x1 + text_w + 10 > img_w:
                    # SHIFT: Move text left to fit
                    shift_amount = (x1 + text_w + 10) - img_w
                    bg_x1 -= shift_amount
                    bg_x2 -= shift_amount
                    text_x -= shift_amount

                # Draw Label Background
                cv2.rectangle(annotated_img, (bg_x1, bg_y1), (bg_x2, bg_y2), self.colors["bg_label"], -1)

                # Draw Text
                cv2.putText(annotated_img, lbl, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors["text"], 2)

        # --- SAVING ---
        if csv_data:
            df = pd.DataFrame(csv_data)
            df = df.sort_values(by="Confidence_Score", ascending=False)
            df.to_csv(os.path.join(csv_dir, f"{base_name}_FULL_REPORT.csv"), index=False)

        cv2.imwrite(os.path.join(visuals_dir, f"{base_name}_ANALYSIS.jpg"), annotated_img)
        print(f" > Processed {base_name}: {len(csv_data)} items logged.")

        return annotated_img, csv_data


# --- EXECUTION ---
if __name__ == "__main__":
    # Ensure you have 'best.pt' and 'yolov8l.pt'
    detector = EnsembleEvidenceDetector(
        standard_weights='yolov8l.pt',
        custom_weights='Custom_Model/weights/best.pt',
    )

    INPUT_FOLDER = "crime_scenes/scene_1.png"

    detector.analyze_crime_scene(INPUT_FOLDER)