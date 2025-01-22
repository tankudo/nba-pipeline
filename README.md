# Next Best Action (NBA) Pipeline

## Overview
The Next Best Action (NBA) system is a machine learning pipeline that predicts optimal actions for user interactions in real-time. It combines historical data analysis with real-time event processing to deliver personalized recommendations.

## Architecture

![image](https://github.com/user-attachments/assets/8181a3a2-21c5-4de3-ad11-6ca363668518)


## Key Features
- Real-time prediction pipeline using LightGBM  
- Feature store with Redis backend  
- Kafka-based event streaming  
- FastAPI REST interface  
- MLflow experiment tracking  
- Comprehensive monitoring and metrics  

## Prerequisites
- Python 3.9+  
- Redis  
- Apache Kafka  
- Docker & Docker Compose  
- AWS CLI (for deployment)  

## Installation


### Clone the repository:
 
```bash
git clone https://github.com/tankudo/nba-pipeline.git
cd nba-pipeline

```
### Use you kaggle API key for dtaset download

1. **Get Kaggle API Key**:
   - Log in to your [Kaggle account](https://www.kaggle.com/).
   - Go to **Account Settings** and click **Create New API Token** to download `kaggle.json`.

2. **Set Up API Key**:
   - Place the `kaggle.json` file in:
     - Windows: `C:\Users\<YourUsername>\.kaggle\`
     - Linux/MacOS: `~/.kaggle/`
   - Ensure proper permissions (Linux/MacOS):
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

### Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
 ### Install dependencies:
 
```bash
pip install -r requirements.txt
```
### Set up configuration:

```bash
cp config/config.yaml.example config/config.yaml
# Edit config.yaml with your settings
```

## Local Development

### Start required services:

```bash
docker-compose up -d redis kafka
```

### Run the API server:

```bash
uvicorn api.app:app --reload --port 8000
```

## Configuration

### Directory Structure Details

![image](https://github.com/user-attachments/assets/72ffc571-7059-430a-a01e-3095076ca6ac)


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request












