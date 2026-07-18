"""
PROJECT 4: AI TEXT RECOGNITION SYSTEM
 - Batch 2026
Using EasyOCR (No Tesseract Required)
"""

import cv2
import numpy as np
from pathlib import Path
import os

# ============================================================================
# FIRST-TIME SETUP
# ============================================================================

print("\n" + "="*70)
print(" OCR System - Initializing".center(70))
print("="*70)

try:
    import easyocr
    print("\n✓ EasyOCR detected")
except ImportError:
    print("\n⚠️  Installing EasyOCR (first time only)...")
    print("   This will download ~500MB of AI models")
    print("   Please wait 2-5 minutes...\n")
    
    import subprocess
    subprocess.check_call(['pip', 'install', 'easyocr'])
    
    import easyocr
    print("\n✓ Installation complete!")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIDENCE_THRESHOLD = 0.80  # 80%

# ============================================================================
# PREPROCESSING MODULE
# ============================================================================

class ImagePreprocessor:
    """Image preprocessing pipeline"""
    
    @staticmethod
    def grayscale(image):
        """Convert to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def denoise(image):
        """Remove noise"""
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    @staticmethod
    def threshold(image):
        """Binary thresholding"""
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    

    
    @staticmethod
    def enhance_contrast(image):
        """Enhance contrast using CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    def process(self, image):
        print("\n[PREPROCESSING PIPELINE]")
        print("  Step 1: Grayscale conversion...")
        gray = self.grayscale(image)
        
        print("  Step 2: Contrast enhancement...")
        enhanced = self.enhance_contrast(gray)
        
        print("  Step 3: Noise reduction...")
        denoised = self.denoise(enhanced)
        
        print("  Step 4: Binary thresholding...")
        thresh = self.threshold(denoised)
        
        print("  [✓] Preprocessing complete\n")
        return thresh, gray

# ============================================================================
# OCR ENGINE
# ============================================================================

class OCREngine:
    def __init__(self):
        print("\n[INITIALIZING OCR ENGINE]")
        print("  Loading AI model (first time: ~2 min)...")
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        self.preprocessor = ImagePreprocessor()
        print("  [✓] OCR Engine ready\n")
    
    def extract_text(self, image_path):
        print("\n" + "="*70)
        print("OPTICAL CHARACTER RECOGNITION".center(70))
        print("="*70)
        
        # Load image
        print(f"\n📂 Loading: {image_path}")
        image = cv2.imread(str(image_path))
        
        if image is None:
            print("❌ Error: Cannot load image")
            return None
        
        h, w = image.shape[:2]
        print(f"   ✓ Dimensions: {w}x{h} pixels")
        
        # Preprocess
        processed, gray = self.preprocessor.process(image)
        
        # OCR Execution
        print("[EASYOCR NEURAL NETWORK RUNNING...]")
        print("  Analyzing image...")
        
        # Run OCR on original and processed
        results_orig = self.reader.readtext(image)
        results_proc = self.reader.readtext(processed)
        
        # Combine and deduplicate
        all_results = results_orig + results_proc
        unique_results = self._deduplicate(all_results)
        
        print(f"   ✓ Detected {len(unique_results)} text segments")
        
        # Filter by confidence
        filtered = [
            {
                'text': text,
                'confidence': float(conf),
                'bbox': bbox
            }
            for bbox, text, conf in unique_results
            if conf >= CONFIDENCE_THRESHOLD
        ]
        
        print(f"   ✓ {len(filtered)} passed confidence threshold (≥{CONFIDENCE_THRESHOLD*100}%)")
        
        return filtered, image, processed
    
    def _deduplicate(self, results):
        seen = set()
        unique = []
        
        for bbox, text, conf in results:
            text_clean = text.strip().lower()
            if text_clean and text_clean not in seen:
                seen.add(text_clean)
                unique.append((bbox, text.strip(), conf))
        
        return unique
    
    def display_results(self, results):
        print("\n" + "="*70)
        print("EXTRACTION RESULTS".center(70))
        print("="*70)
        
        if not results:
            print("\n⚠️  No text detected with sufficient confidence")
            print("\n💡 Suggestions:")
            print("   • Use clearer image with better lighting")
            print("   • Ensure text is horizontal and readable")
            print("   • Try higher resolution image")
            return False
        
        # Calculate statistics
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        
        print(f"\n📊 DETECTION STATISTICS")
        print(f"   Total segments: {len(results)}")
        print(f"   Average confidence: {avg_conf*100:.1f}%")
        print(f"   Min confidence: {min(r['confidence'] for r in results)*100:.1f}%")
        print(f"   Max confidence: {max(r['confidence'] for r in results)*100:.1f}%")
        
        print(f"\n📝 RECOGNIZED TEXT:\n")
        
        # Display with confidence bars
        for idx, r in enumerate(results, 1):
            conf_pct = r['confidence'] * 100
            bar = "█" * int(conf_pct / 10)
            print(f"   {idx:2d}. [{conf_pct:5.1f}%] {bar:10s} {r['text']}")
        
        # Full text reconstruction
        full_text = ' '.join([r['text'] for r in results])
        print(f"\n📄 COMPLETE TEXT:")
        print(f"   {full_text}")
        
        print("\n" + "="*70)
        
        # Validation checkpoint
        if avg_conf >= CONFIDENCE_THRESHOLD:
            print(f"✅ VALIDATION PASSED: {avg_conf*100:.1f}% ≥ {CONFIDENCE_THRESHOLD*100}%")
            return True
        else:
            print(f"⚠️  VALIDATION WARNING: {avg_conf*100:.1f}% < {CONFIDENCE_THRESHOLD*100}%")
            return False
    
    def annotate_image(self, image, results):
        """Draw bounding boxes on detected text"""
        annotated = image.copy()
        
        for r in results:
            bbox = r['bbox']
            conf = r['confidence']
            
            # Convert bbox to integer coordinates
            points = np.array(bbox, dtype=np.int32)
            
            # Color based on confidence
            if conf >= 0.95:
                color = (0, 255, 0)  # Green - Excellent
            elif conf >= CONFIDENCE_THRESHOLD:
                color = (0, 165, 255)  # Orange - Good
            else:
                color = (0, 0, 255)  # Red - Low
            
            # Draw polygon
            cv2.polylines(annotated, [points], True, color, 2)
            
            # Add confidence label
            x, y = int(points[0][0]), int(points[0][1])
            label = f"{conf*100:.0f}%"
            cv2.putText(annotated, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main execution loop"""
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n" + "="*70)
    print("🔬  AI TEXT RECOGNITION SYSTEM".center(70))
    print("Project 4 | Batch 2026 | Powered by EasyOCR".center(70))
    print("="*70)
    
    # Initialize OCR engine
    try:
        engine = OCREngine()
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")
        print("\n🔧 Try running: pip install easyocr")
        input("\nPress Enter to exit...")
        return
    
    print("\n" + "-"*70)
    print("SYSTEM READY".center(70))
    print("-"*70)
    
    while True:
        # Get image path
        print("\n📂 Options:")
        print("   • Enter image file path")
        print("   • Type 'q' to quit")
        
        image_path = input("\n> ").strip().strip('"').strip("'")
        
        if image_path.lower() == 'q':
            break
        
        if not image_path:
            print("⚠️  Please provide an image path")
            continue
        
        # Validate path
        path = Path(image_path)
        if not path.exists():
            print(f"❌ File not found: {image_path}")
            continue
        
        # Execute OCR
        result = engine.extract_text(image_path)
        
        if result is None:
            continue
        
        results, original, processed = result
        
        # Display results
        validation_passed = engine.display_results(results)
        
        # Show images option
        show = input("\n📷 Display processed images? (y/n): ").lower()
        
        if show == 'y' and results:
            # Create annotated version
            annotated = engine.annotate_image(original, results)
            
            # Display
            cv2.imshow('Original Image', original)
            cv2.imshow('Processed Image', processed)
            cv2.imshow('Detected Text (Annotated)', annotated)
            
            print("\n[Press any key in image window to close]")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Save option
        save = input("\n💾 Save results to file? (y/n): ").lower()
        
        if save == 'y':
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            base_name = path.stem
            
            # Save images
            cv2.imwrite(str(output_dir / f"{base_name}_processed.jpg"), processed)
            
            if results:
                annotated = engine.annotate_image(original, results)
                cv2.imwrite(str(output_dir / f"{base_name}_annotated.jpg"), annotated)
                
                # Save text
                with open(output_dir / f"{base_name}_text.txt", 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("EXTRACTED TEXT -  Project 4\n")
                    f.write("="*60 + "\n\n")
                    
                    for i, r in enumerate(results, 1):
                        f.write(f"{i}. [{r['confidence']*100:.1f}%] {r['text']}\n")
                    
                    f.write("\n" + "="*60 + "\n")
                    f.write("FULL TEXT:\n")
                    f.write("="*60 + "\n")
                    full = ' '.join([r['text'] for r in results])
                    f.write(full)
            
            print(f"✓ Results saved to '{output_dir}' folder")
        
        # Continue?
        cont = input("\n🔄 Process another image? (y/n): ").lower()
        if cont != 'y':
            break
    
    print("\n" + "="*70)
    print("Thank you for using  OCR System!".center(70))
    print("Project 4 Complete - Batch 2026".center(70))
    print("="*70 + "\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()