import time
import os
from datetime import timedelta
import torch
import matplotlib.pyplot as plt

def train_loop(train_fn, evaluate_fn, model, train_iterator, validation_iterator, optimizer, criterion, epochs, check_point_path=None):
    best_validation_loss = float('inf')
    train_accuracies = []
    validation_accuraies = []
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_accuracy = train_fn(model, train_iterator, optimizer, criterion)
        validation_loss, validation_accuracy = evaluate_fn(model, validation_iterator, criterion)
        end_time = time.time()

        # display metrics
        if check_point_path and validation_loss < best_validation_loss:
            if not os.path.exists(check_point_path):
                os.makedirs(check_point_path)
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), os.path.join(check_point_path, f"{model.name}-model.pt"))
        elapsed = end_time - start_time
        print(f"Epoch {epoch + 1} | Elapsed: {timedelta(seconds=elapsed)}")
        print(f"Train Loss {train_loss:.2} | Train Accuracy: {train_accuracy:.2%}")
        print(f"Validation Loss {validation_loss:.2} | Validation Accuracy: {validation_accuracy:.2%}\n")
        train_accuracies.append(train_accuracy)
        validation_accuraies.append(validation_accuracy)
    return train_accuracies, validation_accuraies

def plot_learning_curves(train_accuracies, validation_accuraies, title):
    epochs = list(range(1, len(train_accuracies) + 1))
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, validation_accuraies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.show()

def get_accuracy(batch_predictions, batch_labels):

    # put predictions into sigmoid [0, 1] and round to integer
    batch_predictions = torch.round(torch.sigmoid(batch_predictions))
    matches = (batch_predictions == batch_labels).float()
    return matches.sum() / len(matches)

def load_check_point(check_point_path, model):
    model_state_dict = torch.load(os.path.join(check_point_path, f"{model.name}-model.pt"))
    model.load_state_dict(model_state_dict)

def compare_best(check_point_path, model, model2, evaluate_fn, test_iterator, criterion):
    load_check_point(check_point_path, model)
    load_check_point(check_point_path, model2)
    _, validation_accuracy = evaluate_fn(model, test_iterator, criterion)
    _, validation_accuracy2 = evaluate_fn(model2, test_iterator, criterion)
    return validation_accuracy, validation_accuracy2