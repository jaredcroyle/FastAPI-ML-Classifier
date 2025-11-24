# FastAPI ML Classifier for Genetics

A machine learning classifier for genetic data, built with FastAPI and PyTorch. This project demonstrates building and deploying a production-ready ML model with a RESTful API.

## Features

- FastAPI for high-performance API endpoints
- PyTorch for deep learning model implementation
- Docker containerization for easy deployment
- Pandas for data processing
- Pre-trained model included (ONLY FOR SPLICE-JUNCTION GENE SEQUENCE DATA)
(dataset: https://www.kaggle.com/datasets/muhammetvarl/splicejunction-gene-sequences-dataset)

## Prerequisites

- Docker and Docker Compose
- Python 3.10+

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/FastAPI-ML-Classifier.git
   cd FastAPI-ML-Classifier
   ```

2. **Start the application**
   ```bash
   docker compose up --build
   ```

3. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Interactive Docs: http://localhost:8000/redoc

## Project Structure

```
.
├── app/                    # application source code
│   ├── __init__.py
│   ├── main.py            # FastAPI application
│   ├── model.py           # PyTorch model definition
│   ├── encoding.py        # Data preprocessing
│   └── utils.py           # Utility functions
├── train.py               # Model training script
├── test_api.py            # API test script
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
└── requirements.txt       # Python dependencies
```

## API Endpoints

- `POST /predict` - Make predictions with the trained model
- `GET /health` - Check if the API is running

## Development

### Without Docker

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Running the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

## License

[MIT License]

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)
- [Docker](https://www.docker.com/)
