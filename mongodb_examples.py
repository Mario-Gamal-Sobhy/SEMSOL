"""
Example usage of MongoDB integration for SEMSOL engagement monitoring.
Demonstrates how to save sessions and query historical data.
"""

from utils.mongodb_handler import SessionMetricsDB, save_session_to_mongodb
from datetime import datetime, timedelta
import pandas as pd

# ===========================
# Example 1: Quick Save
# ===========================
def example_quick_save():
    """Quick way to save a session after monitoring."""
    
    # Example session data (normally from st.session_state.session_data)
    session_data = {
        'session_start': datetime.now() - timedelta(minutes=10),
        'session_end': datetime.now(),
        'frames_processed': 600,
        'total_blinks': 45,
        'timestamps': [datetime.now() - timedelta(seconds=i) for i in range(10)],
        'engagement_levels': [1, 2, 2, 3, 2, 1, 1, 2, 3, 2],
        'confidence_scores': [0.85, 0.78, 0.82, 0.65, 0.79, 0.88, 0.91, 0.77, 0.68, 0.80],
        'pitch_angles': [5.2, 6.1, 4.8, 12.3, 7.1, 3.9, 4.2, 8.5, 15.2, 6.8],
        'yaw_angles': [2.1, -3.5, 1.8, 8.9, -2.3, 0.5, 1.2, -4.1, 10.5, -1.8],
        'ear_values': [0.28, 0.27, 0.29, 0.26, 0.28, 0.30, 0.29, 0.27, 0.25, 0.28],
        'blink_rates': [0.18, 0.20, 0.19, 0.22, 0.17, 0.18, 0.19, 0.21, 0.23, 0.19],
        'face_detected': [True] * 10,
        'fps_values': [15.2, 14.8, 15.1, 14.9, 15.3, 15.0, 14.7, 15.2, 14.8, 15.1],
        'blink_states': ['normal'] * 8 + ['stressed', 'normal']
    }
    
    # Example session report (normally from generate_session_report())
    session_report = {
        'session_info': {
            'start_time': session_data['session_start'].strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': session_data['session_end'].strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': 600,
            'frames_processed': 600,
            'total_blinks': 45
        },
        'engagement_summary': {
            'highly_engaged_percent': 30.0,
            'engaged_percent': 50.0,
            'partially_engaged_percent': 15.0,
            'disengaged_percent': 5.0,
            'average_confidence': 0.78
        }
    }
    
    # Save to MongoDB
    session_id = save_session_to_mongodb(session_data, session_report)
    print(f"✅ Session saved with ID: {session_id}")


# ===========================
# Example 2: Full Database Operations
# ===========================
def example_full_operations():
    """Demonstrates all database operations."""
    
    # Initialize database connection
    db = SessionMetricsDB(
        connection_string="mongodb://localhost:27017/",
        database_name="semsol_engagement"
    )
    
    # 1. Get recent sessions
    print("\n📋 Recent Sessions:")
    recent = db.get_recent_sessions(limit=5)
    for session in recent:
        print(f"  - {session['session_id']}: {session['start_time']} ({session['frames_processed']} frames)")
    
    # 2. Get a specific session
    if recent:
        session_id = recent[0]['session_id']
        print(f"\n🔍 Session Details for {session_id}:")
        session = db.get_session(session_id)
        print(f"  Duration: {session['duration_seconds']:.1f}s")
        print(f"  Total Blinks: {session['total_blinks']}")
        print(f"  Engagement Summary: {session['summary']['engagement_summary']}")
        
        # 3. Get metrics for this session
        print(f"\n📊 Metrics DataFrame:")
        metrics_df = db.get_session_metrics(session_id)
        print(metrics_df.head())
        print(f"  Total metrics: {len(metrics_df)}")
    
    # 4. Get sessions by date range
    print("\n📅 Sessions from last 7 days:")
    week_ago = datetime.now() - timedelta(days=7)
    sessions = db.get_sessions_by_date_range(week_ago, datetime.now())
    print(f"  Found {len(sessions)} sessions")
    
    # 5. Get aggregate statistics
    print("\n📈 Aggregate Engagement Statistics:")
    stats = db.get_engagement_statistics()
    print(f"  Total frames analyzed: {stats.get('total_frames', 0)}")
    for level, data in stats.get('engagement_distribution', {}).items():
        print(f"  Level {level}: {data['percentage']:.1f}% (confidence: {data['avg_confidence']:.2f})")
    
    # Close connection
    db.close()


# ===========================
# Example 3: Historical Analysis
# ===========================
def example_historical_analysis():
    """Analyze trends across multiple sessions."""
    
    db = SessionMetricsDB()
    
    # Get all sessions from the last month
    month_ago = datetime.now() - timedelta(days=30)
    sessions = db.get_sessions_by_date_range(month_ago, datetime.now())
    
    if not sessions:
        print("No sessions found in the last month")
        db.close()
        return
    
    print(f"\n📊 Analyzing {len(sessions)} sessions from the last month\n")
    
    # Analyze engagement trends
    session_ids = [s['session_id'] for s in sessions]
    
    # Create a DataFrame for analysis
    all_data = []
    for session_id in session_ids:
        metrics = db.get_session_metrics(session_id)
        if not metrics.empty:
            all_data.append(metrics)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print("📈 Overall Statistics:")
        print(f"  Total frames: {len(combined_df)}")
        print(f"  Average engagement level: {combined_df['engagement_level'].mean():.2f}")
        print(f"  Average confidence: {combined_df['confidence_score'].mean():.2%}")
        print(f"  Average blink rate: {combined_df['blink_rate'].mean():.3f} bps")
        
        print("\n🎯 Engagement Distribution:")
        engagement_dist = combined_df['engagement_level'].value_counts(normalize=True).sort_index()
        labels = {1: "Highly Engaged", 2: "Engaged", 3: "Partially Engaged", 4: "Disengaged"}
        for level, pct in engagement_dist.items():
            print(f"  {labels.get(level, f'Level {level}')}: {pct:.1%}")
        
        print("\n👁️ Blink State Distribution:")
        blink_dist = combined_df['blink_state'].value_counts(normalize=True)
        for state, pct in blink_dist.items():
            print(f"  {state.capitalize()}: {pct:.1%}")
    
    db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("SEMSOL MongoDB Integration Examples")
    print("=" * 60)
    
    # Uncomment the example you want to run:
    
    # example_quick_save()
    # example_full_operations()
    # example_historical_analysis()
    
    print("\n✅ Examples completed!")
