"""
Add test session data to MongoDB for demonstration.
Run this to populate your database with sample engagement monitoring sessions.
"""

from utils.mongodb_handler import SessionMetricsDB
from datetime import datetime, timedelta
import random

def generate_test_session(session_number):
    """Generate a realistic test session."""
    
    # Session timing
    start_time = datetime.now() - timedelta(hours=session_number * 2)
    duration = random.randint(300, 900)  # 5-15 minutes
    end_time = start_time + timedelta(seconds=duration)
    
    # Generate metrics (one per second)
    num_frames = duration
    timestamps = [start_time + timedelta(seconds=i) for i in range(num_frames)]
    
    # Simulate realistic engagement patterns
    engagement_levels = []
    confidence_scores = []
    pitch_angles = []
    yaw_angles = []
    ear_values = []
    blink_rates = []
    blink_states = []
    
    for i in range(num_frames):
        # Engagement varies over time (students get tired)
        time_factor = i / num_frames
        if time_factor < 0.3:  # First 30% - highly engaged
            engagement_levels.append(random.choices([1, 2], weights=[0.7, 0.3])[0])
        elif time_factor < 0.7:  # Middle 40% - engaged
            engagement_levels.append(random.choices([1, 2, 3], weights=[0.2, 0.6, 0.2])[0])
        else:  # Last 30% - getting tired
            engagement_levels.append(random.choices([2, 3, 4], weights=[0.3, 0.5, 0.2])[0])
        
        confidence_scores.append(random.uniform(0.65, 0.95))
        pitch_angles.append(random.uniform(-10, 15))
        yaw_angles.append(random.uniform(-12, 12))
        ear_values.append(random.uniform(0.24, 0.32))
        blink_rates.append(random.uniform(0.15, 0.25))
        blink_states.append(random.choices(
            ['normal', 'drowsy', 'stressed', 'distracted'],
            weights=[0.7, 0.1, 0.1, 0.1]
        )[0])
    
    # Calculate summary statistics
    total_blinks = int(sum(blink_rates) * duration / 60)  # Approximate
    
    engagement_counts = {
        1: engagement_levels.count(1),
        2: engagement_levels.count(2),
        3: engagement_levels.count(3),
        4: engagement_levels.count(4)
    }
    
    session_data = {
        'session_start': start_time,
        'session_end': end_time,
        'frames_processed': num_frames,
        'total_blinks': total_blinks,
        'timestamps': timestamps,
        'engagement_levels': engagement_levels,
        'confidence_scores': confidence_scores,
        'pitch_angles': pitch_angles,
        'yaw_angles': yaw_angles,
        'ear_values': ear_values,
        'blink_rates': blink_rates,
        'face_detected': [True] * num_frames,
        'fps_values': [random.uniform(14.5, 15.5) for _ in range(num_frames)],
        'blink_states': blink_states
    }
    
    session_report = {
        'session_info': {
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': duration,
            'frames_processed': num_frames,
            'total_blinks': total_blinks
        },
        'engagement_summary': {
            'highly_engaged_percent': (engagement_counts[1] / num_frames) * 100,
            'engaged_percent': (engagement_counts[2] / num_frames) * 100,
            'partially_engaged_percent': (engagement_counts[3] / num_frames) * 100,
            'disengaged_percent': (engagement_counts[4] / num_frames) * 100,
            'average_confidence': sum(confidence_scores) / len(confidence_scores)
        },
        'gaze_summary': {
            'avg_pitch': sum(pitch_angles) / len(pitch_angles),
            'std_pitch': 5.2,
            'avg_yaw': sum(yaw_angles) / len(yaw_angles),
            'std_yaw': 4.8
        },
        'blink_summary': {
            'avg_ear': sum(ear_values) / len(ear_values),
            'avg_blink_rate': sum(blink_rates) / len(blink_rates),
            'blink_states': {
                'normal': blink_states.count('normal'),
                'drowsy': blink_states.count('drowsy'),
                'stressed': blink_states.count('stressed'),
                'distracted': blink_states.count('distracted')
            }
        },
        'performance': {
            'avg_fps': 15.0
        }
    }
    
    return session_data, session_report


def main():
    print("=" * 70)
    print("🧪 Adding Test Data to MongoDB")
    print("=" * 70)
    
    try:
        # Connect to MongoDB
        db = SessionMetricsDB(
            connection_string="mongodb://localhost:27017/",
            database_name="semsol_engagement"
        )
        print("✅ Connected to MongoDB\n")
        
        # Generate and save 3 test sessions
        num_sessions = 3
        print(f"Generating {num_sessions} test sessions...\n")
        
        for i in range(num_sessions):
            print(f"Creating session {i+1}/{num_sessions}...")
            session_data, session_report = generate_test_session(i)
            
            session_id = db.save_session(session_data, session_report)
            print(f"  ✅ Saved: {session_id}")
            print(f"     Duration: {session_report['session_info']['duration_seconds']}s")
            print(f"     Frames: {session_report['session_info']['frames_processed']}")
            print(f"     Avg Engagement: {session_report['engagement_summary']['highly_engaged_percent']:.1f}% highly engaged")
            print()
        
        # Show summary
        print("=" * 70)
        print("📊 Database Summary")
        print("-" * 70)
        total_sessions = db.sessions_collection.count_documents({})
        total_metrics = db.metrics_collection.count_documents({})
        print(f"Total Sessions: {total_sessions}")
        print(f"Total Metrics: {total_metrics:,}")
        
        db.close()
        
        print("\n" + "=" * 70)
        print("✅ Test data added successfully!")
        print("=" * 70)
        print("\n📍 VIEW IN MONGODB COMPASS:")
        print("   1. Open MongoDB Compass")
        print("   2. Connect to: mongodb://localhost:27017")
        print("   3. Click on database: semsol_engagement")
        print("   4. View collections:")
        print("      - sessions (session metadata)")
        print("      - metrics (detailed frame-by-frame data)")
        print("\n💡 TIP: Run 'python view_mongodb_data.py' to view in terminal")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure MongoDB is running")
        print("2. Check if MongoDB Compass can connect to localhost:27017")
        print("3. Verify pymongo is installed: pip install pymongo")


if __name__ == "__main__":
    main()
