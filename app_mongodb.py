"""
app_mongodb.py - Enhanced Streamlit App with MongoDB Integration
==================================================================

This is a wrapper that adds MongoDB functionality to your existing app.py.
It modifies the behavior to auto-save sessions to MongoDB.

USAGE:
    streamlit run app_mongodb.py

This will run your app with MongoDB integration enabled.
"""

import sys
import os

# Add MongoDB integration by monkey-patching
def integrate_mongodb():
    """Add MongoDB support to the app."""
    try:
        from utils.mongodb_handler import save_session_to_mongodb
        from mongodb_config import MONGODB_CONFIG
        import streamlit as st
        
        # Store original rerun
        original_rerun = st.rerun
        
        def mongodb_aware_rerun():
            """Enhanced rerun that saves to MongoDB when stopping."""
            # Check if we're stopping monitoring
            if hasattr(st.session_state, 'monitoring'):
                if not st.session_state.monitoring and \
                   hasattr(st.session_state, 'session_data') and \
                   st.session_state.session_data.get('session_end') is not None and \
                   not hasattr(st.session_state, '_mongodb_saved'):
                    
                    # Save to MongoDB
                    try:
                        # Import here to avoid circular imports
                        from datetime import datetime
                        import pandas as pd
                        
                        # Generate report
                        session_data = st.session_state.session_data
                        
                        # Simple report generation
                        df = pd.DataFrame({
                            'timestamp': session_data.get('timestamps', []),
                            'engagement_level': session_data.get('engagement_levels', []),
                            'confidence': session_data.get('confidence_scores', []),
                        })
                        
                        if len(df) > 0:
                            session_report = {
                                'session_info': {
                                    'start_time': session_data['session_start'].strftime('%Y-%m-%d %H:%M:%S'),
                                    'end_time': session_data['session_end'].strftime('%Y-%m-%d %H:%M:%S'),
                                    'duration_seconds': (session_data['session_end'] - session_data['session_start']).total_seconds(),
                                    'frames_processed': session_data.get('frames_processed', 0),
                                    'total_blinks': session_data.get('total_blinks', 0)
                                },
                                'engagement_summary': {
                                    'highly_engaged_percent': (df['engagement_level'] == 1).sum() / len(df) * 100,
                                    'engaged_percent': (df['engagement_level'] == 2).sum() / len(df) * 100,
                                    'partially_engaged_percent': (df['engagement_level'] == 3).sum() / len(df) * 100,
                                    'disengaged_percent': (df['engagement_level'] == 4).sum() / len(df) * 100,
                                    'average_confidence': df['confidence'].mean()
                                },
                                'gaze_summary': {},
                                'blink_summary': {},
                                'performance': {}
                            }
                            
                            session_id = save_session_to_mongodb(
                                session_data,
                                session_report,
                                MONGODB_CONFIG['connection_string']
                            )
                            
                            st.session_state._mongodb_saved = True
                            st.session_state.last_saved_session_id = session_id
                            print(f"✅ Session saved to MongoDB: {session_id}")
                    
                    except Exception as e:
                        print(f"⚠️ MongoDB save failed: {e}")
            
            # Call original rerun
            original_rerun()
        
        # Replace rerun
        st.rerun = mongodb_aware_rerun
        print("✅ MongoDB integration enabled")
        
    except ImportError as e:
        print(f"⚠️ MongoDB not available: {e}")

# Integrate before importing app
integrate_mongodb()

# Now run the original app
if __name__ == "__main__":
    # Import and run the original app
    import runpy
    runpy.run_path("app.py", run_name="__main__")
