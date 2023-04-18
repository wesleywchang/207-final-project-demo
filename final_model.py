import numpy as np
import pandas as pd
# import tflite_runtime.interpreter as tflite
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# tokenization
with open('tokens.json', 'r') as f:
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

word_index = tokenizer.word_index

def pad_data(sequences, max_length):
  # Keras has a convenient utility for padding a sequence.
  # Also make sure we get a numpy array rather than an array of lists.
  return np.array(list(
      tf.keras.preprocessing.sequence.pad_sequences(
          sequences, maxlen=max_length, padding='post', value=0)))

def limit_vocab(sequences, max_token_id, oov_id=2):
  """Replace token ids greater than or equal to max_token_id with the oov_id."""
  reduced_sequences = np.copy(sequences)
  reduced_sequences[reduced_sequences >= max_token_id] = oov_id
  return reduced_sequences

def preprocess(sequences, max_length, vocab_size):
  sequences = tokenizer.texts_to_sequences(sequences)
  return limit_vocab(pad_data(sequences, max_length=max_length), max_token_id=vocab_size)
  
vocab_size = 1500
epochs = 10
embedding_dim = 128
sequence_length = 40

def predict_text(input):
    input_data_raw = [input,]
    input_data = preprocess(input_data_raw, max_length=sequence_length,vocab_size=vocab_size).astype('float32')
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.round(np.squeeze(output_data),3)

    decode = ['negative', 'neutral', 'positive']
    
    sentiment = decode[np.argmax(results)]

    return([sentiment, results])