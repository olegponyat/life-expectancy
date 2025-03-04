import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

x = pd.read_csv('quotes.csv')
x['quote'] = x['quote'].fillna('').astype(str)

MAX_QUOTE_LENGTH = 200
x['quote'] = x['quote'].apply(lambda q: q[:MAX_QUOTE_LENGTH] if len(q) > MAX_QUOTE_LENGTH else q)

all_characters = string.ascii_lowercase + ' '
char_to_index = {char: idx for idx, char in enumerate(all_characters)}
index_to_char = {idx: char for idx, char in enumerate(all_characters)}

def get_letter_frequencies(quote):
    letter_count = {char: 0 for char in all_characters}
    quote = quote.lower().replace(' ', '') 

    for char in quote:
        if char in all_characters:
            letter_count[char] += 1
    
    total_chars = sum(letter_count.values())
    normalized_frequencies = {char: count / total_chars if total_chars > 0 else 0 for char, count in letter_count.items()}
    
    return np.array([normalized_frequencies[char] for char in all_characters])

def quote_to_targets(quote):
    letter_count = {char: 0 for char in all_characters}
    quote = quote.lower().replace(' ', '') 
    
    for char in quote:
        if char in all_characters:
            letter_count[char] += 1
    
    total_letters = sum(letter_count.values())
    letter_frequencies = np.array([letter_count[char] / total_letters if total_letters > 0 else 0 for char in all_characters])
    
    return letter_frequencies

def generate_batch_data(batch_size=1000):
    batch_X = []
    batch_y = []
    
    for i in range(0, len(x), batch_size):
        batch_X_batch = []
        batch_y_batch = []
        for quote in x['quote'][i:i+batch_size]:
            letter_frequencies = get_letter_frequencies(quote)
            targets = quote_to_targets(quote)
            
            batch_X_batch.append(letter_frequencies)
            batch_y_batch.append(targets)
        
        batch_X.append(np.array(batch_X_batch))
        batch_y.append(np.array(batch_y_batch))
    
    return np.vstack(batch_X), np.vstack(batch_y)

X, y = generate_batch_data(batch_size=1000)

print(f"Processed X shape: {X.shape}")
print(f"Processed y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, input_dim=27, activation='relu'),
    Dense(128, activation='relu'),
    Dense(27, activation='softmax'), 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_loss(history)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

def reconstruct_quote(predictions, max_length=200):
    reconstructed_quote = []

    for i in range(min(len(predictions), max_length)):
        letter_idx = np.argmax(predictions[i]) 
        
        if letter_idx < 27:
            reconstructed_quote.append(index_to_char[letter_idx])
        else:
            reconstructed_quote.append(' ')
    
    return ''.join(reconstructed_quote).strip()


test_quote = "The only thing we have to fear is fear itself"
test_sequence = get_letter_frequencies(test_quote)
test_sequence_padded = np.array([test_sequence])
predicted_frequencies = model.predict(test_sequence_padded)

reconstructed_quote = reconstruct_quote(predicted_frequencies[0])

print("Original Quote:", test_quote)
print("Reconstructed Quote:", reconstructed_quote)
