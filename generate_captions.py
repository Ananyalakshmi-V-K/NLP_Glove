"""
Glove caption generator.
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class GloveFeatureExtractor:
    """Feature extractor."""
    
    def __init__(self):
        # BGR color defs
        self.color_ranges = {
            'black': ([0, 0, 0], [180, 255, 50]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'gray': ([0, 0, 50], [180, 50, 200]),
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'brown': ([10, 50, 50], [20, 255, 200]),
            'orange': ([10, 100, 100], [25, 255, 255]),
            'pink': ([140, 50, 50], [170, 255, 255]),
        }
        
        # size thresholds
        self.size_thresholds = {
            'small': (0, 0.25),
            'medium': (0.25, 0.45),
            'large': (0.45, 0.70),
            'extra large': (0.70, 1.0)
        }
    
    def extract_color(self, image_path):
        """Detect color."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return "unknown"
            
            # convert HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            color_counts = {}
            for color_name, (lower, upper) in self.color_ranges.items():
                lower = np.array(lower)
                upper = np.array(upper)
                mask = cv2.inRange(hsv, lower, upper)
                color_counts[color_name] = cv2.countNonZero(mask)
            
            # get dominant
            dominant_color = max(color_counts.items(), key=lambda x: x[1])
            
            # fallback RGB
            if dominant_color[1] < 1000:
                avg_color = cv2.mean(img)[:3]
                if avg_color[0] < 50 and avg_color[1] < 50 and avg_color[2] < 50:
                    return "black"
                elif avg_color[0] > 200 and avg_color[1] > 200 and avg_color[2] > 200:
                    return "white"
                else:
                    return "multicolor"
            
            return dominant_color[0]
        
        except Exception as e:
            print(f"Error extracting color from {image_path}: {e}")
            return "unknown"
    
    def estimate_size(self, image_path):
        """Estimate size."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return "medium"
            
            # to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # threshold image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                image_area = img.shape[0] * img.shape[1]
                coverage = contour_area / image_area
                
                # classify size
                for size, (min_cov, max_cov) in self.size_thresholds.items():
                    if min_cov <= coverage < max_cov:
                        return size
            
            return "medium"
        
        except Exception as e:
            print(f"Error estimating size from {image_path}: {e}")
            return "medium"
    
    def detect_pattern(self, image_path):
        """Detect pattern."""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return "plain"
            
            # resize image
            img = cv2.resize(img, (224, 224))
            
            # edge density
            edges = cv2.Canny(img, 50, 150)
            edge_density = np.sum(edges > 0) / (224 * 224)
            
            # std deviation
            std_dev = np.std(img)
            
            # classify pattern
            if edge_density > 0.15 and std_dev > 40:
                return "textured"
            elif edge_density > 0.10:
                return "patterned"
            elif std_dev > 50:
                return "dotted"
            else:
                return "plain"
        
        except Exception as e:
            print(f"Error detecting pattern from {image_path}: {e}")
            return "plain"
    
    def detect_material(self, image_path):
        """Detect material."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return "fabric"
            
            # to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # calc smoothness
            variance = np.var(gray)
            
            # calc brightness
            brightness = np.mean(gray)
            
            # classify material
            if variance < 500 and brightness > 100:
                return "rubber"
            elif variance < 1000:
                return "leather"
            elif variance > 2000:
                return "knitted"
            else:
                return "fabric"
        
        except Exception as e:
            print(f"Error detecting material from {image_path}: {e}")
            return "fabric"
    
    def detect_defects(self, image_path):
        """Detect defects."""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return "no visible damage"
            
            # blob detection
            # blob params
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 50
            params.maxArea = 5000
            
            # init detector
            detector = cv2.SimpleBlobDetector_create(params)
            
            # detect blobs
            keypoints = detector.detect(img)
            
            if len(keypoints) > 10:
                return "multiple marks or stains"
            elif len(keypoints) > 5:
                return "some visible marks"
            else:
                return "no visible damage"
        
        except Exception as e:
            print(f"Error detecting defects from {image_path}: {e}")
            return "no visible damage"
    
    def extract_all_features(self, image_path):
        """Extract all features."""
        features = {
            'color': self.extract_color(image_path),
            'size': self.estimate_size(image_path),
            'pattern': self.detect_pattern(image_path),
            'material': self.detect_material(image_path),
            'condition': self.detect_defects(image_path)
        }
        return features


class CaptionGenerator:
    """Caption generator."""
    
    def __init__(self):
        self.templates = [
            "{size} {color} {material} glove with {pattern} pattern, {condition}",
            "{color} {material} glove, size {size}, featuring {pattern} design, {condition}",
            "{size} sized {color} glove made of {material}, {pattern} texture, {condition}",
            "{material} glove in {color} color, {size} size, {pattern} finish, {condition}",
        ]
    
    def generate_caption(self, features, template_idx=0):
        """Make caption."""
        template = self.templates[template_idx % len(self.templates)]
        
        try:
            caption = template.format(**features)
            # capitalize
            caption = caption[0].upper() + caption[1:]
            # add period
            if not caption.endswith('.'):
                caption += '.'
            return caption
        except KeyError as e:
            print(f"Missing feature: {e}")
            return f"{features.get('color', 'unknown')} glove."


def generate_captions_for_dataset(image_dir, output_file):
    """Generate all captions."""
    
    print("Initializing feature extractor...")
    extractor = GloveFeatureExtractor()
    generator = CaptionGenerator()
    
    # find images
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} images")
    
    captions_data = []
    
    print("\nExtracting features and generating captions...")
    for idx, img_file in enumerate(tqdm(image_files)):
        img_path = os.path.join(image_dir, img_file)
        
        # get features
        features = extractor.extract_all_features(img_path)
        
        # gen caption
        caption = generator.generate_caption(features, template_idx=idx)
        
        # build entry
        entry = {
            "image": img_file,
            "caption": caption,
            "features": features
        }
        
        captions_data.append(entry)
    
    # save JSON
    with open(output_file, 'w') as f:
        json.dump(captions_data, f, indent=2)
    
    print(f"\nGenerated {len(captions_data)} captions")
    print(f"Saved to {output_file}")
    
    # show samples
    print("\nSample captions:")
    for i in range(min(5, len(captions_data))):
        print(f"\n{i+1}. Image: {captions_data[i]['image']}")
        print(f"   Caption: {captions_data[i]['caption']}")
        print(f"   Features: {captions_data[i]['features']}")


if __name__ == "__main__":
    IMAGE_DIR = "data/images"
    OUTPUT_FILE = "data/captions.json"
    
    generate_captions_for_dataset(IMAGE_DIR, OUTPUT_FILE)
