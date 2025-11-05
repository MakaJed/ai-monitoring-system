# Model Training Guide - Adding Negative Examples

## Overview
To reduce false positives (like detecting ears as chickens), you need to add **negative examples** to your training dataset. Negative examples are images with **no annotations** that teach the model what NOT to detect.

## Why Negative Examples Matter
- **Without negatives**: Model may detect anything that vaguely looks like a chicken (ears, shadows, objects)
- **With negatives**: Model learns to distinguish between actual chickens and false positives

## How to Add Negative Examples

### For Chicken Detection Model (`chickenmodel.tflite`)

1. **Collect negative images:**
   - Images with no chickens visible
   - Images with objects that might be confused (ears, shadows, background objects)
   - Images from different angles, lighting conditions, backgrounds

2. **Annotation format:**
   - For images with NO chickens: Create annotation file with **empty annotations** or **no file at all**
   - Most YOLO training tools accept empty annotation files (just the file exists but has no boxes)
   - Example: `image001.jpg` → `image001.txt` (empty file or file with just "0" class with no boxes)

3. **YOLO format example:**
   ```
   # For positive example (has chicken):
   # image001.txt contains:
   0 0.5 0.5 0.3 0.4  # class x_center y_center width height (normalized)
   
   # For negative example (no chicken):
   # image002.txt contains:
   # (empty file or no file)
   ```

4. **Training dataset structure:**
   ```
   dataset/
   ├── images/
   │   ├── chicken001.jpg  (positive)
   │   ├── chicken002.jpg  (positive)
   │   ├── negative001.jpg (negative - no annotations)
   │   ├── negative002.jpg (negative - no annotations)
   │   └── ...
   └── labels/
       ├── chicken001.txt  (has annotations)
       ├── chicken002.txt  (has annotations)
       ├── negative001.txt (empty or doesn't exist)
       └── negative002.txt (empty or doesn't exist)
   ```

5. **Recommended ratio:**
   - **Positive:Negative ratio**: 1:1 to 1:2 (for every 1 chicken image, add 1-2 negative images)
   - Start with 1:1, increase negatives if still getting false positives

### For Symptoms Model (`ndvsymptoms.tflite`)

Same approach:
- Collect images of healthy chickens (no symptoms visible)
- Add as negative examples (no symptom annotations)
- Helps model distinguish between actual symptoms and normal chicken features

## Current Detection Settings

The detection thresholds have been adjusted to reduce false positives:

### Chicken Detection:
- **Confidence**: 0.5 (was 0.35) - Higher = fewer false positives
- **Min Area**: 0.01 ratio (was 0.003) - Filters tiny detections
- **Aspect Ratio**: 0.4 to 2.5 (width/height) - Filters non-chicken shapes

### Symptoms Detection:
- **Confidence**: 0.45 (was 0.35) - Higher = fewer false positives

## Adjusting Thresholds

You can adjust these via environment variables:

```bash
# For chicken detection
export DETECT_CONF=0.5           # Confidence threshold (0.0-1.0, higher = stricter)
export MIN_AREA_RATIO=0.01       # Minimum box area (0.0-1.0, higher = filter smaller boxes)
export MIN_ASPECT_RATIO=0.4      # Minimum width/height ratio
export MAX_ASPECT_RATIO=2.5      # Maximum width/height ratio

# For symptoms detection
export SYMPTOMS_CONF=0.45        # Confidence threshold
```

Or in your systemd service file:
```ini
Environment="DETECT_CONF=0.5"
Environment="MIN_AREA_RATIO=0.01"
Environment="SYMPTOMS_CONF=0.45"
```

## Testing After Training

1. **Test with known false positives:**
   - Images where ears were detected as chickens
   - Images with shadows/background objects
   
2. **Monitor detection logs:**
   - Check `/detect` endpoint responses
   - Look at saved images in `data/logs/YYYY-MM-DD/`
   - Review bounding boxes in the dashboard

3. **Adjust thresholds if needed:**
   - If too many false positives: Increase confidence threshold
   - If missing real chickens: Decrease confidence threshold
   - If detecting tiny objects: Increase MIN_AREA_RATIO

## Tips for Better Training

1. **Diversity matters:**
   - Different lighting conditions (day/night, shadows)
   - Different angles (side, front, back)
   - Different backgrounds (coop, ground, different textures)
   - Different chicken poses (standing, sitting, walking)

2. **Quality over quantity:**
   - 100 well-annotated images > 1000 poorly annotated ones
   - Make sure bounding boxes are accurate
   - Include edge cases (partially visible chickens, unusual angles)

3. **Balance your dataset:**
   - Don't have all images from one angle
   - Mix positive and negative examples
   - Include challenging cases

## Next Steps

1. Collect negative examples (images with no chickens)
2. Add them to your training dataset with empty annotations
3. Retrain the model with the balanced dataset
4. Test and adjust confidence thresholds as needed

