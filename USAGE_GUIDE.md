# Helmet Detection Main Application Usage Guide

This guide explains how to use the `main.py` application for helmet detection and tracking.

## Quick Start

### Basic Usage (Webcam)
```bash
python main.py
```

### With Custom Model
```bash
python main.py --model /path/to/your/model.pt
```

### Process Video File
```bash
python main.py --source /path/to/video.mp4
```

## Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--source` | `-s` | `0` | Video source (0 for webcam, or path to video file) |
| `--model` | `-m` | `best.pt` | Path to YOLOv5 model weights |
| `--conf` | `-c` | `0.2` | Confidence threshold (0.0-1.0) |
| `--iou` | `-i` | `0.0` | IoU threshold for helmet detection |
| `--max-age` | | `30` | Maximum age for track persistence |
| `--device` | `-d` | `auto` | Device to run inference on (auto/cuda/cpu) |
| `--output` | `-o` | `None` | Output video file path |
| `--no-display` | | | Disable video display (headless mode) |
| `--save-stats` | | `None` | Save statistics to CSV file |

## Examples

### 1. Webcam with Custom Settings
```bash
python main.py --conf 0.3 --iou 0.5 --device cuda
```

### 2. Process Video File and Save Output
```bash
python main.py --source input_video.mp4 --output output_video.mp4
```

### 3. Headless Processing (No Display)
```bash
python main.py --source video.mp4 --no-display --save-stats stats.csv
```

### 4. High Confidence Detection
```bash
python main.py --conf 0.5 --iou 0.7
```

### 5. CPU Processing
```bash
python main.py --device cpu
```

## Keyboard Controls

When the application is running, you can use these keyboard shortcuts:

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `r` | Reset the tracker |
| `s` | Show current statistics |
| `h` | Show help |
| `p` | Pause/Resume processing |

## Output Features

### Real-time Display
- Bounding boxes around detected people
- Color coding: Green = Helmet, Red = No Helmet
- Track ID and status labels
- Real-time statistics overlay

### Statistics
The application tracks and displays:
- Number of persons detected
- People with helmets
- People without helmets
- Active tracks
- Frame processing rate (FPS)

### CSV Export
When using `--save-stats`, the application saves detailed statistics including:
- Frame number
- Timestamp
- Detection counts
- Track information

## Performance Tips

1. **Use GPU**: Set `--device cuda` for faster processing
2. **Adjust Confidence**: Lower confidence for more detections, higher for accuracy
3. **IoU Threshold**: Adjust based on your helmet detection needs
4. **Headless Mode**: Use `--no-display` for faster processing without GUI

## Troubleshooting

### Common Issues

1. **Model not found**
   ```bash
   python main.py --model /correct/path/to/model.pt
   ```

2. **CUDA out of memory**
   ```bash
   python main.py --device cpu
   ```

3. **Poor detection quality**
   ```bash
   python main.py --conf 0.1 --iou 0.3
   ```

4. **Video file not found**
   - Ensure the video file path is correct
   - Check file permissions

### Error Messages

- **"Could not open video source"**: Check if webcam is available or video file exists
- **"Error initializing HelmetTracker"**: Check model path and dependencies
- **"Could not create output video file"**: Check write permissions and disk space

## Integration

The `main.py` application uses the `HelmetTracker` module, so you can also:

1. Import the module in your own scripts
2. Use the module's API directly
3. Extend functionality by modifying the module

## Requirements

Make sure you have installed all required dependencies:
```bash
pip install torch torchvision opencv-python deep-sort-realtime ultralytics
``` 