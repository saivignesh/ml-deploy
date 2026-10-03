# Sentiment Prediction

Trained a logistic regression model on the reviews dataset, used in the workshop,
to predict whether a review text is positive or not.

# API endpoints

`GET /health` -> `{ "status" : "ok"}` 

`POST /predict` -> `{ "sentiment" : "Positive" , "confidence": 0.9383 }`

# Run Locally

`uv run uvicorn main:app --host 0.0.0.0 --port 8000` 
