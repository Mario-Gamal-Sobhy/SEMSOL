"""
MongoDB Handler for Session Metrics Storage
Stores engagement monitoring session data in MongoDB for historical analysis.
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionMetricsDB:
    """
    Handles MongoDB operations for storing and retrieving session metrics.
    """
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", 
                 database_name: str = "semsol_engagement"):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Name of the database to use
        """
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.server_info()
            self.db = self.client[database_name]
            self.sessions_collection = self.db['sessions']
            self.metrics_collection = self.db['metrics']
            
            # Create indexes for efficient querying
            self._create_indexes()
            
            logger.info(f"✅ Connected to MongoDB: {database_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for efficient querying."""
        # Index on session_id for fast lookups
        self.sessions_collection.create_index([("session_id", ASCENDING)])
        self.sessions_collection.create_index([("start_time", DESCENDING)])
        
        # Index on session_id and timestamp for metrics
        self.metrics_collection.create_index([
            ("session_id", ASCENDING),
            ("timestamp", ASCENDING)
        ])
    
    def save_session(self, session_data: Dict, session_report: Dict) -> str:
        """
        Save a complete session to MongoDB.
        
        Args:
            session_data: Dictionary containing session metrics (from st.session_state)
            session_report: Dictionary containing session summary statistics
            
        Returns:
            session_id: Unique identifier for the saved session
        """
        try:
            # Generate unique session ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare session document
            session_doc = {
                "session_id": session_id,
                "start_time": session_data['session_start'],
                "end_time": session_data['session_end'],
                "duration_seconds": (session_data['session_end'] - session_data['session_start']).total_seconds(),
                "frames_processed": session_data['frames_processed'],
                "total_blinks": session_data['total_blinks'],
                "summary": session_report,
                "created_at": datetime.now()
            }
            
            # Insert session metadata
            self.sessions_collection.insert_one(session_doc)
            
            # Prepare and insert detailed metrics
            metrics_docs = []
            for i in range(len(session_data['timestamps'])):
                metric_doc = {
                    "session_id": session_id,
                    "timestamp": session_data['timestamps'][i],
                    "engagement_level": session_data['engagement_levels'][i],
                    "confidence_score": session_data['confidence_scores'][i],
                    "pitch_angle": session_data['pitch_angles'][i],
                    "yaw_angle": session_data['yaw_angles'][i],
                    "ear_value": session_data['ear_values'][i],
                    "blink_rate": session_data['blink_rates'][i],
                    "face_detected": session_data['face_detected'][i],
                    "fps": session_data['fps_values'][i],
                    "blink_state": session_data['blink_states'][i]
                }
                metrics_docs.append(metric_doc)
            
            # Batch insert metrics
            if metrics_docs:
                self.metrics_collection.insert_many(metrics_docs)
            
            logger.info(f"✅ Saved session {session_id} with {len(metrics_docs)} metrics")
            return session_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save session: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session document or None if not found
        """
        try:
            session = self.sessions_collection.find_one(
                {"session_id": session_id},
                {"_id": 0}  # Exclude MongoDB's internal ID
            )
            return session
        except Exception as e:
            logger.error(f"❌ Failed to retrieve session {session_id}: {e}")
            return None
    
    def get_session_metrics(self, session_id: str) -> pd.DataFrame:
        """
        Retrieve all metrics for a specific session as a DataFrame.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            DataFrame containing all metrics for the session
        """
        try:
            metrics = list(self.metrics_collection.find(
                {"session_id": session_id},
                {"_id": 0}  # Exclude MongoDB's internal ID
            ).sort("timestamp", ASCENDING))
            
            if not metrics:
                logger.warning(f"No metrics found for session {session_id}")
                return pd.DataFrame()
            
            return pd.DataFrame(metrics)
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve metrics for {session_id}: {e}")
            return pd.DataFrame()
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent sessions.
        
        Args:
            limit: Maximum number of sessions to retrieve
            
        Returns:
            List of session documents
        """
        try:
            sessions = list(self.sessions_collection.find(
                {},
                {"_id": 0}
            ).sort("start_time", DESCENDING).limit(limit))
            
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve recent sessions: {e}")
            return []
    
    def get_sessions_by_date_range(self, start_date: datetime, 
                                   end_date: datetime) -> List[Dict]:
        """
        Get sessions within a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of session documents
        """
        try:
            sessions = list(self.sessions_collection.find(
                {
                    "start_time": {
                        "$gte": start_date,
                        "$lte": end_date
                    }
                },
                {"_id": 0}
            ).sort("start_time", DESCENDING))
            
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve sessions by date range: {e}")
            return []
    
    def get_engagement_statistics(self, session_ids: Optional[List[str]] = None) -> Dict:
        """
        Calculate aggregate engagement statistics across sessions.
        
        Args:
            session_ids: List of session IDs to analyze (None = all sessions)
            
        Returns:
            Dictionary containing aggregate statistics
        """
        try:
            query = {}
            if session_ids:
                query["session_id"] = {"$in": session_ids}
            
            pipeline = [
                {"$match": query},
                {"$group": {
                    "_id": "$engagement_level",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence_score"}
                }}
            ]
            
            results = list(self.metrics_collection.aggregate(pipeline))
            
            # Format results
            stats = {
                "engagement_distribution": {},
                "total_frames": 0
            }
            
            for result in results:
                level = result["_id"]
                count = result["count"]
                stats["engagement_distribution"][level] = {
                    "count": count,
                    "avg_confidence": result["avg_confidence"]
                }
                stats["total_frames"] += count
            
            # Calculate percentages
            for level in stats["engagement_distribution"]:
                stats["engagement_distribution"][level]["percentage"] = (
                    stats["engagement_distribution"][level]["count"] / stats["total_frames"] * 100
                )
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate engagement statistics: {e}")
            return {}
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its metrics.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete session metadata
            self.sessions_collection.delete_one({"session_id": session_id})
            
            # Delete all metrics for this session
            self.metrics_collection.delete_many({"session_id": session_id})
            
            logger.info(f"✅ Deleted session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete session {session_id}: {e}")
            return False
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Convenience function for quick session saving
def save_session_to_mongodb(session_data: Dict, session_report: Dict,
                           connection_string: str = "mongodb://localhost:27017/") -> Optional[str]:
    """
    Quick function to save a session to MongoDB.
    
    Args:
        session_data: Session data from st.session_state
        session_report: Session report from generate_session_report()
        connection_string: MongoDB connection URI
        
    Returns:
        session_id if successful, None otherwise
    """
    try:
        db = SessionMetricsDB(connection_string)
        session_id = db.save_session(session_data, session_report)
        db.close()
        return session_id
    except Exception as e:
        logger.error(f"Failed to save session: {e}")
        return None
