"""
app_mongodb_integration.py

This file contains the code snippets to add MongoDB integration to app.py.
Copy and paste these sections into your app.py file.
"""

# ============================================================================
# SECTION 1: Add these imports at the top of app.py (after existing imports)
# ============================================================================

from utils.mongodb_handler import SessionMetricsDB, save_session_to_mongodb
from mongodb_config import MONGODB_CONFIG, AUTO_SAVE_CONFIG

# ============================================================================
# SECTION 2: Add MongoDB settings to sidebar (after existing sidebar controls)
# ============================================================================

st.sidebar.divider()
st.sidebar.markdown("### 🗄️ Database Settings")

enable_mongodb = st.sidebar.checkbox(
    "Enable MongoDB Storage", 
    value=AUTO_SAVE_CONFIG['enabled'],
    help="Automatically save sessions to MongoDB"
)

if enable_mongodb:
    mongodb_status = st.sidebar.empty()
    
    # Test MongoDB connection
    try:
        test_db = SessionMetricsDB(
            connection_string=MONGODB_CONFIG['connection_string'],
            database_name=MONGODB_CONFIG['database_name']
        )
        test_db.close()
        mongodb_status.success("✅ MongoDB Connected")
    except Exception as e:
        mongodb_status.error(f"❌ MongoDB Error: {str(e)[:50]}")
        enable_mongodb = False

# ============================================================================
# SECTION 3: Replace the "Stop Monitoring" button logic (around line 187)
# ============================================================================

# OLD CODE:
"""
if st.sidebar.button("⏹ Stop Monitoring", type="secondary", use_container_width=True):
    st.session_state.monitoring = False
    st.session_state.session_data['session_end'] = datetime.now()
    st.rerun()
"""

# NEW CODE:
if st.sidebar.button("⏹ Stop Monitoring", type="secondary", use_container_width=True):
    st.session_state.monitoring = False
    st.session_state.session_data['session_end'] = datetime.now()
    
    # Auto-save to MongoDB if enabled
    if enable_mongodb:
        try:
            # Generate visualizations and report
            visualizations = generate_session_visualizations(st.session_state.session_data)
            if visualizations:
                session_report = generate_session_report(
                    st.session_state.session_data, 
                    visualizations
                )
                
                # Save to MongoDB
                session_id = save_session_to_mongodb(
                    st.session_state.session_data,
                    session_report,
                    MONGODB_CONFIG['connection_string']
                )
                
                # Store session_id for later reference
                st.session_state.last_saved_session_id = session_id
                st.sidebar.success(f"✅ Saved: {session_id}")
                
        except Exception as e:
            st.sidebar.error(f"⚠️ DB save failed: {e}")
    
    st.rerun()

# ============================================================================
# SECTION 4: Add Session History Tab (replace main content area)
# ============================================================================

# Create tabs for Live Monitoring and History
tab1, tab2 = st.tabs(["🎥 Live Monitoring", "📊 Session History"])

with tab1:
    # ALL EXISTING MONITORING CODE GOES HERE
    # (Everything from "if st.session_state.monitoring:" onwards)
    pass

with tab2:
    st.subheader("📊 Session History")
    
    if not enable_mongodb:
        st.warning("⚠️ Enable MongoDB in sidebar to view session history")
    else:
        try:
            db = SessionMetricsDB(
                connection_string=MONGODB_CONFIG['connection_string'],
                database_name=MONGODB_CONFIG['database_name']
            )
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                num_sessions = st.number_input("Sessions to show", 1, 50, 10)
            with col2:
                date_filter = st.date_input("From date", datetime.now() - timedelta(days=7))
            with col3:
                if st.button("🔄 Refresh"):
                    st.rerun()
            
            # Get sessions
            sessions = db.get_recent_sessions(limit=num_sessions)
            
            if not sessions:
                st.info("No sessions found in database")
            else:
                st.success(f"Found {len(sessions)} sessions")
                
                # Display sessions
                for session in sessions:
                    with st.expander(
                        f"📅 {session['session_id']} - "
                        f"{session['start_time'].strftime('%Y-%m-%d %H:%M:%S')} "
                        f"({session['duration_seconds']:.0f}s)"
                    ):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Session Info**")
                            st.write(f"Frames: {session['frames_processed']}")
                            st.write(f"Blinks: {session['total_blinks']}")
                            st.write(f"Duration: {session['duration_seconds']:.1f}s")
                        
                        with col2:
                            st.markdown("**Engagement Summary**")
                            summary = session['summary']['engagement_summary']
                            st.write(f"Highly Engaged: {summary['highly_engaged_percent']:.1f}%")
                            st.write(f"Engaged: {summary['engaged_percent']:.1f}%")
                            st.write(f"Partially Engaged: {summary['partially_engaged_percent']:.1f}%")
                            st.write(f"Disengaged: {summary['disengaged_percent']:.1f}%")
                        
                        # View detailed metrics
                        if st.button("📊 View Detailed Metrics", key=f"view_{session['session_id']}"):
                            metrics_df = db.get_session_metrics(session['session_id'])
                            
                            if not metrics_df.empty:
                                st.dataframe(metrics_df, use_container_width=True)
                                
                                # Download button
                                csv = metrics_df.to_csv(index=False)
                                st.download_button(
                                    "⬇️ Download CSV",
                                    csv,
                                    f"{session['session_id']}.csv",
                                    "text/csv",
                                    key=f"download_{session['session_id']}"
                                )
                        
                        # Delete session
                        if st.button("🗑️ Delete Session", key=f"delete_{session['session_id']}"):
                            if db.delete_session(session['session_id']):
                                st.success("Session deleted")
                                st.rerun()
            
            # Aggregate statistics
            st.divider()
            st.subheader("📈 Aggregate Statistics")
            
            stats = db.get_engagement_statistics()
            if stats:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Frames Analyzed", stats.get('total_frames', 0))
                
                with col2:
                    # Create pie chart
                    if 'engagement_distribution' in stats:
                        labels = ['Highly Engaged', 'Engaged', 'Partially Engaged', 'Disengaged']
                        values = [
                            stats['engagement_distribution'].get(i, {}).get('count', 0) 
                            for i in [1, 2, 3, 4]
                        ]
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            marker=dict(colors=['#00ff00', '#ffff00', '#ffa500', '#ff0000'])
                        )])
                        fig.update_layout(title='Overall Engagement Distribution', height=300)
                        st.plotly_chart(fig, use_container_width=True)
            
            db.close()
            
        except Exception as e:
            st.error(f"Error loading session history: {e}")

# ============================================================================
# SECTION 5: Add download session report after monitoring stops
# ============================================================================

# Add this in the "else" block when monitoring is stopped
if not st.session_state.monitoring and st.session_state.session_data['session_end'] is not None:
    st.divider()
    st.subheader("📊 Session Complete")
    
    # Show last saved session ID
    if hasattr(st.session_state, 'last_saved_session_id'):
        st.success(f"✅ Session saved to database: {st.session_state.last_saved_session_id}")
    
    # Generate and display visualizations
    visualizations = generate_session_visualizations(st.session_state.session_data)
    
    if visualizations:
        session_report = generate_session_report(
            st.session_state.session_data,
            visualizations
        )
        
        # Display visualizations
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(visualizations['engagement_timeline'], use_container_width=True)
            st.plotly_chart(visualizations['blink_analysis'], use_container_width=True)
        
        with col2:
            st.plotly_chart(visualizations['gaze_heatmap'], use_container_width=True)
            st.plotly_chart(visualizations['engagement_pie'], use_container_width=True)
        
        # Download options
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download CSV
            csv = visualizations['dataframe'].to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv,
                "session_metrics.csv",
                "text/csv"
            )
        
        with col2:
            # Download JSON report
            import json
            json_report = json.dumps(session_report, indent=2, default=str)
            st.download_button(
                "⬇️ Download Report (JSON)",
                json_report,
                "session_report.json",
                "application/json"
            )
        
        with col3:
            # Manual save to DB (if not auto-saved)
            if not enable_mongodb:
                if st.button("💾 Save to Database"):
                    try:
                        session_id = save_session_to_mongodb(
                            st.session_state.session_data,
                            session_report,
                            MONGODB_CONFIG['connection_string']
                        )
                        st.success(f"✅ Saved: {session_id}")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
