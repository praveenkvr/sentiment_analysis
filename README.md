# Sentiment Analysis

## url: https://sentimentml.com

Model has been trained on a twitter dataset and a attention based architecture has been used to build the model.

# ML

To train the model run

`python train.py`

To Predict on text

`python main.py --text "this is positive"`

Unfortunately i can not upload the trained weights in this repo due to github repo size limits.

# UI

UI code has been developed using nextjs. and can be run locally using the following command

`cd ui/ && npm run dev`

# server

Server has been implemented in python and flask. To run the server locally

`python api.py`

make sure the model weigts and the tokenizer.pickl are available in the src folder

# TODO

The model will continue to evolve over time and will be training the model over different sentiment datasets so it could provide accurate results.
