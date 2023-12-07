# Sentiment Analysis Project

This project focuses on a comparative analysis between Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) for sentiment analysis, utilizing the Sentiment140 dataset from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140/).

## Getting Started

### Python Virtual Environment

Set up a python virtual environment:

```shell
python -m venv <venv>
```

Activate the environment:

```shell
<venv>/Scripts/activate
```

Install requirements:

```shell
pip install -r requirements.txt
```

> Note: To record your environment's current package list into `requirements.txt`:
>
> ```shell
> pip freeze > requirements.txt
> ```

## Implementation Details

### Models Overview

#### RNN Models

- RNN architecture with an embedding layer, two RNN layers, a dropout layer, and a fully connected layer.
- LSTM architecture with an embedding layer, two bidirectional LSTM layers, a dropout layer, and a fully connected layer.

#### CNN Models

- CNN architecture with an embedding layer, a convolutional layer, ReLU activation, max pooling, dropout, and a fully connected layer.
- A more complex CNN architecture with additional convolutional layers.

### Training and Results

The models were trained on the Sentiment140 dataset and evaluated based on test accuracy. The CNN models demonstrated superior generalization compared to the RNN counterparts. The LSTM model, in particular, outperformed other models.

### Future Work

Future work could include optimizing model configurations, exploring ensemble models, incorporating pre-trained language models, and extending the study to diverse domains or languages. Additionally, investigating interpretability methods for deep learning models could provide insights into the decision-making processes.