import redis
import json
import yaml

class FeatureStore:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.redis_client = redis.Redis(
            host=self.config['feature_store']['host'],
            port=self.config['feature_store']['port']
        )
    
    def get_user_features(self, user_id):
        """Get user features from Redis"""
        features = self.redis_client.get(f"user:{user_id}")
        return json.loads(features) if features else None
    
    def update_features(self, user_id, features):
        """Update user features in Redis"""
        self.redis_client.set(f"user:{user_id}", json.dumps(features))