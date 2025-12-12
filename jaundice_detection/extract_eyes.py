"""
Eye Region Extraction Script
Extracts eye regions from face images using OpenCV.
"""

import cv2
import os
import argparse
from pathlib import Path


def extract_eyes(
    input_dir: str = 'raw',
    output_dir: str = 'raw_eyes',
    eye_size: int = 224
):
    """
    Extract eye regions from face images.
    
    Args:
        input_dir: Directory containing face images (with jaundice/normal subdirs)
        output_dir: Output directory for extracted eyes
        eye_size: Output size for eye images
    """
    
    # Load Haar cascades
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    classes = [d.name for d in input_path.iterdir() if d.is_dir()]
    
    for cls in classes:
        class_input = input_path / cls
        class_output = output_path / cls
        class_output.mkdir(parents=True, exist_ok=True)
        
        images = list(class_input.glob('*.jpg')) + list(class_input.glob('*.png'))
        
        extracted = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect face
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                # Try without face detection
                eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
            else:
                # Look for eyes within face region
                x, y, w, h = faces[0]
                face_gray = gray[y:y+h, x:x+w]
                face_color = img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)
                
                if len(eyes) > 0:
                    # Extract eye region
                    ex, ey, ew, eh = eyes[0]
                    eye_img = face_color[ey:ey+eh, ex:ex+ew]
                    
                    # Resize
                    eye_img = cv2.resize(eye_img, (eye_size, eye_size))
                    
                    # Save
                    output_name = f"{img_path.stem}_eye.jpg"
                    cv2.imwrite(str(class_output / output_name), eye_img)
                    extracted += 1
                    continue
            
            # Fallback: use center of image as eye region estimate
            h, w = img.shape[:2]
            center_y = int(h * 0.35)  # Eyes typically in upper third
            crop_size = min(w, h) // 3
            
            eye_region = img[
                center_y - crop_size//2:center_y + crop_size//2,
                w//2 - crop_size//2:w//2 + crop_size//2
            ]
            
            if eye_region.size > 0:
                eye_region = cv2.resize(eye_region, (eye_size, eye_size))
                output_name = f"{img_path.stem}_eye.jpg"
                cv2.imwrite(str(class_output / output_name), eye_region)
                extracted += 1
        
        print(f"{cls}: Extracted {extracted}/{len(images)} eye regions")
    
    print(f"\n✓ Eye regions saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract eye regions')
    parser.add_argument('--input-dir', type=str, default='raw')
    parser.add_argument('--output-dir', type=str, default='raw_eyes')
    parser.add_argument('--eye-size', type=int, default=224)
    
    args = parser.parse_args()
    
    extract_eyes(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        eye_size=args.eye_size
    )
