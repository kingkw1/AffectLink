def video_processing_loop(video_emotions, video_lock, stop_flag, video_started_event):
    """Thread for processing video frames for emotion detection"""
    global CAMERA_INDEX
    
    print(f"Initializing video capture with camera index: {CAMERA_INDEX}")
    
    # Import the camera utility module
    try:
        from camera_utils import find_available_camera
        
        # Use our camera utility to find an available camera
        preferred_index = CAMERA_INDEX
        use_directshow = os.environ.get('WEBCAM_BACKEND', '').lower() == 'directshow'
        
        print(f"Looking for camera (preferred index: {preferred_index}, directshow: {use_directshow})")
        camera_idx, backend, cap = find_available_camera(
            preferred_index=preferred_index, 
            use_directshow=use_directshow
        )
        
        if cap is None:
            print("Error: Cannot access any webcam. Continuing with audio-only analysis.")
            # Signal that video processing has attempted to start (even if failed)
            video_started_event.set()
            
            # Don't stop the entire system, just exit this thread
            while not stop_flag['stop']:
                time.sleep(1)  # Keep thread alive but idle
            return
    
    except Exception as e:
        print(f"Error initializing camera: {e}")
        # Signal that video processing has attempted to start (even if failed)
        video_started_event.set()
        
        # Don't stop the entire system, just exit this thread
        while not stop_flag['stop']:
            time.sleep(1)  # Keep thread alive but idle
        return
        
    # Signal that video processing has started
    video_started_event.set()
    
    # For 5-second downsampling
    window_start_time = time.time()
    frame_emotions = []
    
    retry_count = 0
    max_retries = 5
    
    while not stop_flag['stop']:
        try:
            ret, frame = cap.read()
            if not ret:
                retry_count += 1
                print(f"Error: Failed to read video frame. Retry {retry_count}/{max_retries}...")
                if retry_count >= max_retries:
                    print("Maximum retries reached. Continuing with audio-only analysis.")
                    # Keep thread alive but idle instead of exiting completely
                    while not stop_flag['stop']:
                        time.sleep(1)
                    break
                time.sleep(2)
                continue
                
            # Reset retry count on successful frame read
            retry_count = 0
            
            # Process this frame with DeepFace
            results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            faces = results if isinstance(results, list) else [results]
            
            for face in faces:
                if 'dominant_emotion' in face:
                    current_time = time.time()
                    emo = face['dominant_emotion']
                    confidence = face.get('emotion', {}).get(emo, None)
                    emotion_scores = face.get('emotion', {})
                    
                    # Store this frame's emotion data for the current window
                    frame_emotions.append({
                        'timestamp': current_time,
                        'emotion': emo,
                        'confidence': confidence,
                        'emotion_scores': emotion_scores
                    })
                    
                    # Draw rectangle and overlay text
                    region = face.get('region', {})
                    x, y, w, h = region.get('x',0), region.get('y',0), region.get('w',0), region.get('h',0)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                    text_y = y-10 if y-10>10 else y+h+20
                    cv2.putText(frame, f"{emo}", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                else:
                    print("No face detected or emotion data unavailable.")
                    
            # Check if we've reached the end of a 5-second window
            current_time = time.time()
            if current_time - window_start_time >= VIDEO_WINDOW_DURATION and frame_emotions:
                # Calculate average scores for each emotion category
                unified_emotion_scores = {emotion: 0.0 for emotion in UNIFIED_EMOTIONS}
                count = 0
                
                for frame_data in frame_emotions:
                    raw_scores = frame_data.get('emotion_scores', {})
                    # Map DeepFace emotions to our unified set and accumulate scores
                    if 'neutral' in raw_scores:
                        unified_emotion_scores['neutral'] += raw_scores.get('neutral', 0)
                    if 'happy' in raw_scores:
                        unified_emotion_scores['happy'] += raw_scores.get('happy', 0)
                    if 'sad' in raw_scores:
                        unified_emotion_scores['sad'] += raw_scores.get('sad', 0)
                    if 'angry' in raw_scores:
                        unified_emotion_scores['angry'] += raw_scores.get('angry', 0)
                    count += 1
                
                if count > 0:
                    # Average the scores
                    for emotion in unified_emotion_scores:
                        unified_emotion_scores[emotion] /= count
                    
                    # Find the dominant emotion from the averaged scores
                    dominant_emotion = max(unified_emotion_scores.items(), key=lambda x: x[1])
                    dominant_label = dominant_emotion[0]
                    dominant_score = dominant_emotion[1]
                    
                    # Create the aggregated video emotion entry
                    aggregated_entry = {
                        'timestamp': current_time,  # End of the 5-second window
                        'emotion': dominant_label,
                        'confidence': dominant_score,
                        'emotion_scores': unified_emotion_scores
                    }
                    
                    # Add to video emotions log with lock
                    with video_lock:
                        video_emotions.append(aggregated_entry)
                
                # Reset for next window
                window_start_time = current_time
                frame_emotions = []
                
            cv2.imshow('Real-time Video Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag['stop'] = True
                break
                
        except Exception as e:
            print(f"Video analysis error: {e}")
            time.sleep(0.1)  # Prevent tight loop in case of repeated errors
            
    # Clean up resources
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
