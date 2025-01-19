import pandas as pd
from kafka import KafkaConsumer
import yaml
import json

class DataIngestion:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.consumer = KafkaConsumer(
            self.config['kafka_config']['topic'],
            bootstrap_servers=self.config['kafka_config']['bootstrap_servers'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
    
    def ingest_batch_data(self, filepath):
        """Ingest historical batch data"""
        return pd.read_csv(filepath)
    
    def stream_events(self):
        """Stream real-time events from Kafka"""
        for message in self.consumer:
            yield message.value