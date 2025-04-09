import cv2
import numpy as np
import os
from pathlib import Path
import shutil

def analyze_cloud_properties(image):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate cloud coverage
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        coverage = (np.sum(thresh == 255) / (thresh.shape[0] * thresh.shape[1])) * 100
        
        # Calculate cloud density
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        density = (np.sum(mask == 255) / (mask.shape[0] * mask.shape[1])) * 100
        
        return coverage, density
    except Exception as e:
        print(f"Error in analyze_cloud_properties: {str(e)}")
        return 0, 0

def is_suitable_for_seeding(coverage, density):
    # Criteria for suitable clouds:
    # Coverage should be at least 55%
    return coverage >= 55.0

def analyze_dataset():
    try:
        print("Starting cloud analysis...")
        
        # Create directories for sorted images
        print("Creating output directories...")
        os.makedirs("Ai_vision_model/cloudDataset/sorted_clouds/suitable", exist_ok=True)
        os.makedirs("Ai_vision_model/cloudDataset/sorted_clouds/not_suitable", exist_ok=True)
        
        # Get all images from both directories
        print("Looking for image directories...")
        cloud_dir = Path("Ai_vision_model/cloudDataset/cloud")
        not_cloud_dir = Path("Ai_vision_model/cloudDataset/not_cloud")
        
        print(f"Cloud directory exists: {cloud_dir.exists()}")
        print(f"Not cloud directory exists: {not_cloud_dir.exists()}")
        
        if not cloud_dir.exists() or not not_cloud_dir.exists():
            print("Error: Dataset directories not found!")
            print("Please make sure the following directories exist:")
            print("- Ai_vision_model/cloudDataset/cloud")
            print("- Ai_vision_model/cloudDataset/not_cloud")
            return
        
        # Combine all images
        print("Scanning for images...")
        cloud_images = list(cloud_dir.glob("*.jpg"))
        not_cloud_images = list(not_cloud_dir.glob("*.jpg"))
        all_images = cloud_images + not_cloud_images
        
        print(f"Found {len(cloud_images)} cloud images")
        print(f"Found {len(not_cloud_images)} not-cloud images")
        print(f"Total images to analyze: {len(all_images)}")
        
        if not all_images:
            print("No images found in the dataset directories!")
            return
        
        print("\nStarting automatic image analysis based on coverage criteria...\n")
        
        suitable_count = 0
        not_suitable_count = 0
        
        for i, img_path in enumerate(all_images, 1):
            print(f"\nProcessing image {i}/{len(all_images)}: {img_path.name}")
            img = cv2.imread(str(img_path))
            if img is not None:
                print(f"Image loaded successfully. Size: {img.shape}")
                coverage, density = analyze_cloud_properties(img)
                print(f"Analysis complete - Coverage: {coverage:.1f}%, Density: {density:.1f}%")
                
                if is_suitable_for_seeding(coverage, density):
                    dest = os.path.join("Ai_vision_model/cloudDataset/sorted_clouds/suitable", img_path.name)
                    shutil.copy2(str(img_path), dest)
                    print(f"Image {img_path.name} moved to suitable directory (Coverage: {coverage:.1f}%)")
                    suitable_count += 1
                else:
                    dest = os.path.join("Ai_vision_model/cloudDataset/sorted_clouds/not_suitable", img_path.name)
                    shutil.copy2(str(img_path), dest)
                    print(f"Image {img_path.name} moved to not_suitable directory (Coverage: {coverage:.1f}%)")
                    not_suitable_count += 1
            else:
                print(f"Warning: Could not read image {img_path}")
        
        print("\nAnalysis complete!")
        print(f"Total images processed: {suitable_count + not_suitable_count}")
        print(f"Images marked as suitable: {suitable_count}")
        print(f"Images marked as not suitable: {not_suitable_count}")
        print(f"\nImages have been sorted into:")
        print(f"- Ai_vision_model/cloudDataset/sorted_clouds/suitable/")
        print(f"- Ai_vision_model/cloudDataset/sorted_clouds/not_suitable/")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    print("Starting automatic cloud analysis...")
    analyze_dataset() 