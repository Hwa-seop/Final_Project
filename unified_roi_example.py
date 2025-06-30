#!/usr/bin/env python3
"""
Example usage of the UnifiedROITracker module.

This script demonstrates how to use the UnifiedROITracker class for
combined helmet detection and ROI-based tracking.
"""

import cv2
import time
from unified_roi_tracker_module import UnifiedROITracker

def main():
    """Main function demonstrating unified ROI tracking."""
    
    # Initialize the unified ROI tracker
    print("Initializing UnifiedROITracker...")
    tracker = UnifiedROITracker(
        model_path='/home/lws/kulws2025/kubig2025/final_project/yolov5/helmet_detection/helmet_detection/weights/best.pt',
        conf_thresh=0.2,
        max_age=30,
        device='auto'  # Will auto-detect CUDA/CPU
    )
    
    # Open video capture (0 for webcam, or provide video file path)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    
    # Setup ROI interactively
    print("Setting up ROI...")
    if not tracker.setup_roi(cap):
        print("ROI setup failed or cancelled")
        cap.release()
        return
    
    print("Starting unified ROI tracking... Press 'q' to quit, 'r' to reset tracker")
    
    # Performance tracking
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            # Process frame
            processed_frame, stats = tracker.process_frame(frame, draw_detections=True)
            
            if processed_frame is not None:
                # Display statistics on frame
                stats_text = [
                    f"Persons: {stats.get('persons_detected', 0)}",
                    f"With Helmet: {stats.get('people_with_helmets', 0)}",
                    f"Without Helmet: {stats.get('people_without_helmets', 0)}",
                    f"In Danger Zone: {stats.get('people_in_danger_zone', 0)}",
                    f"Active Tracks: {stats.get('tracks_active', 0)}"
                ]
                
                y_offset = 30
                for text in stats_text:
                    cv2.putText(processed_frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                
                # Show frame
                cv2.imshow("Unified ROI Tracking", processed_frame)
            
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                print("Resetting tracker...")
                tracker.reset_tracker()
            elif key == ord('s'):
                # Print current statistics
                overall_stats = tracker.get_statistics()
                print("\n=== Current Statistics ===")
                for key, value in overall_stats.items():
                    print(f"{key}: {value}")
                print("========================\n")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
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

if __name__ == "__main__":
    main() 