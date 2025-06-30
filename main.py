#!/usr/bin/env python3
"""
Main application for helmet detection and tracking.

This script provides a complete helmet detection system using the HelmetTracker module.
It can process video from webcam, video files, or image sequences.
"""

import cv2
import argparse
import time
import sys
import os
from helmet_tracking_module import HelmetTracker
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Helmet Detection and Tracking System')
    
    parser.add_argument('--source', '-s', type=str, default='0',
                       help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--model', '-m', type=str, 
                       default='/home/lws/kulws2025/kubig2025/final_project/yolov5/helmet_detection/helmet_detection/weights/best.pt',
                       help='Path to YOLOv5 model weights')
    parser.add_argument('--conf', '-c', type=float, default=0.2,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--iou', '-i', type=float, default=0.0,
                       help='IoU threshold for helmet detection')
    parser.add_argument('--max-age', type=int, default=30,
                       help='Maximum age for track persistence')
    parser.add_argument('--device', '-d', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display (useful for headless operation)')
    parser.add_argument('--save-stats', type=str, default=None,
                       help='Save statistics to CSV file')
    
    return parser.parse_args()

def setup_video_capture(source):
    """Setup video capture from source."""
    if source.isdigit():
        # Webcam
        cap = cv2.VideoCapture(int(source))
    else:
        # Video file
        if not os.path.exists(source):
            print(f"Error: Video file '{source}' not found")
            return None
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'")
        return None
    
    return cap

def setup_video_writer(cap, output_path):
    """Setup video writer for output."""
    if output_path is None:
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default FPS for webcam
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print(f"Error: Could not create output video file '{output_path}'")
        return None
    
    return writer

def save_statistics(stats_list, filename):
    """Save statistics to CSV file."""
    import csv
    
    if not stats_list:
        return
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = stats_list[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for stats in stats_list:
            writer.writerow(stats)
    
    print(f"Statistics saved to {filename}")

def display_help():
    """Display keyboard controls help."""
    print("\n=== Keyboard Controls ===")
    print("q: Quit")
    print("r: Reset tracker")
    print("s: Show statistics")
    print("h: Show this help")
    print("p: Pause/Resume")
    print("=======================\n")

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Display help
    display_help()
    
    # Setup video capture
    print(f"Opening video source: {args.source}")
    cap = setup_video_capture(args.source)
    if cap is None:
        sys.exit(1)
    
    # Setup video writer
    writer = setup_video_writer(cap, args.output)
    
    # Initialize helmet tracker
    print("Initializing HelmetTracker...")
    try:
        tracker = HelmetTracker(
            model_path=args.model,
            conf_thresh=args.conf,
            max_age=args.max_age,
            device=args.device
        )
        print("HelmetTracker initialized successfully!")
    except Exception as e:
        print(f"Error initializing HelmetTracker: {e}")
        cap.release()
        sys.exit(1)
    
    # Statistics tracking
    stats_list = []
    start_time = time.time()
    frame_count = 0
    paused = False
    
    print("Starting helmet detection...")
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                # Process frame
                processed_frame, frame_stats = tracker.process_frame(
                    frame, 
                    draw_detections=True,
                    iou_threshold=args.iou
                )
                
                if processed_frame is not None:
                    # Add frame number and timestamp
                    frame_stats['frame_number'] = frame_count
                    frame_stats['timestamp'] = int(time.time() - start_time)
                    stats_list.append(frame_stats)
                    
                    # Display statistics on frame
                    stats_text = [
                        f"Frame: {frame_count}",
                        f"Persons: {frame_stats.get('persons_detected', 0)}",
                        f"With Helmet: {frame_stats.get('people_with_helmets', 0)}",
                        f"Without Helmet: {frame_stats.get('people_without_helmets', 0)}",
                        f"Active Tracks: {frame_stats.get('tracks_active', 0)}"
                    ]
                    
                    y_offset = 30
                    for text in stats_text:
                        cv2.putText(processed_frame, text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        y_offset += 25
                    
                    # Add pause indicator
                    if paused:
                        cv2.putText(processed_frame, "PAUSED", (10, y_offset + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # Write to output video
                    if writer:
                        writer.write(processed_frame)
                    
                    # Display frame
                    if not args.no_display:
                        cv2.imshow("Helmet Detection", processed_frame)
                
                frame_count += 1
            else:
                # When paused, still process key events
                if not args.no_display and processed_frame is not None:
                    cv2.imshow("Helmet Detection", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                print("Resetting tracker...")
                tracker.reset_tracker()
                stats_list = []  # Reset statistics
            elif key == ord('s'):
                # Print current statistics
                overall_stats = tracker.get_statistics()
                print("\n=== Current Statistics ===")
                for key, value in overall_stats.items():
                    print(f"{key}: {value}")
                print("========================\n")
            elif key == ord('h'):
                display_help()
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        # Print final statistics
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n=== Final Statistics ===")
        print(f"Frames processed: {frame_count}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Average FPS: {fps:.2f}")
        
        overall_stats = tracker.get_statistics()
        for key, value in overall_stats.items():
            print(f"{key}: {value}")
        
        # Save statistics if requested
        if args.save_stats and stats_list:
            save_statistics(stats_list, args.save_stats)

if __name__ == "__main__":
    main()
