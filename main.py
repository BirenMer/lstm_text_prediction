import os
import pickle
from LSTM_text_prediction.model_utils import load_model, save_model
from prediction_function import generate_text
from run_LSTM import prepare_text_data, RunMyLSTM


def main(file_path, seq_length=100, n_neurons=256, n_epoch=2, batch_size=1024, model_path="saved_model"):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    model_file = os.path.join(model_path, "model.pkl")
    if os.path.exists(model_file):
        print("Loading existing model...")
        return load_model(model_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    
    X, Y, char_to_idx, idx_to_char = prepare_text_data(text, seq_length)
    
    lstm, dense_layers, _, _ = RunMyLSTM(
        X, Y,
        vocab_size=len(char_to_idx),
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        n_epoch=n_epoch,
        n_neurons=n_neurons,
        batch_size=batch_size
    )
    
    save_model(lstm, dense_layers, char_to_idx, idx_to_char, model_path)
    return lstm, dense_layers, char_to_idx, idx_to_char

if __name__ == "__main__":
    lstm, dense_layers, char_to_idx, idx_to_char = main("/home/wifee/workspace/learning/ML_book/LSTM_text_prediction/Mobi_dick_book.txt")

    seed_text = "Here they saw such huge troops of whales,".lower()
    print("Available characters:", char_to_idx.keys())
    
    generated_text = generate_text(lstm, dense_layers, seed_text, char_to_idx, idx_to_char, length=500)
    print("\nGenerated Text:\n")
    print(generated_text)