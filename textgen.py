import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
#from tensorflow.keras.layers import LayerNormalization
#from keras import preprocessing
import numpy as np
import os
import time

path_to_file = r"D:\Free_time_learning\DesktopApp\CalendarGUI\tral\dossier\text_combined.txt"

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')

# Take a look at the first 250 characters in text
print(text[:250])

# The unique characters in the file
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

#The preprocessing.StringLookup layer can convert each character into a numeric ID. It just needs the text to be split into tokens first.
example_texts = ['abcdefg', 'xyz']
chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')

#Now create the preprocessing.StringLookup layer:
ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)

# it converts form tokens to character IDs:
#ids = ids_from_chars(chars)
# <tf.RaggedTensor [[40, 41, 42, 43, 44, 45, 46], [63, 64, 65]]>

#Since the goal of this tutorial is to generate text,
#it will also be important to invert this representation and recover human-readable strings from it.
#For this you can use preprocessing.StringLookup(..., invert=True).


chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

#You can tf.strings.reduce_join to join the characters back into strings.

#tf.strings.reduce_join(chars, axis=-1).numpy()

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

#TRAINING EXAMPLES AND TARGETS

#break the text into chunks of seq_length+1. For example, say seq_length is 4 and our text is "Hello". The input sequence would be "Hell",
# and the target sequence "ello".

#To do this first use the tf.data.Dataset.from_tensor_slices function to convert the text vector into a stream of character indices.

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
#all_ids
# <tf.Tensor: shape=(1115394,), dtype=int64, numpy=array([19, 48, 57, ..., 46,  9,  1])>

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))


seq_length = 100 ##100
examples_per_epoch = len(text)//(seq_length+1)

# The batch method lets you easily convert these individual characters to sequences of the desired size.

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(1):
  print(chars_from_ids(seq))


# see what this is doing if you join the tokens back into strings:
for seq in sequences.take(5):
  print(text_from_ids(seq).numpy())


# For training you'll need a dataset of (input, label) pairs. Where input and label are sequences. 
# At each time step the input is the current character and the label is the next character.
# Here's a function that takes a sequence as input, duplicates, and shifts it to align the input and label for each timestep:

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target) ## ??

# showing how the input and target examples are made
for input_example, target_example in dataset.take(1):
    print("Input :", text_from_ids(input_example).numpy())
    print("Target:", text_from_ids(target_example).numpy())

# Create training batches
# You used tf.data to split the text into manageable sequences. 
# But before feeding this data into the model, you need to shuffle the data and pack it into batches.

# Batch size
BATCH_SIZE = 64 #64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 1000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

# Build The Model

#This section defines the model as a keras.Model subclass (For details see Making new Layers and Models via subclassing).

#This model has three layers:

# tf.keras.layers.Embedding: The input layer. A trainable lookup table that will map each character-ID to a vector with embedding_dim dimensions;
# tf.keras.layers.GRU: A type of RNN with size units=rnn_units (You can also use an LSTM layer here.)
# tf.keras.layers.Dense: The output layer, with vocab_size outputs. It outputs one logit for each character in the vocabulary. These are the log-likelihood of each character according to the model.


# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256 ##256

# Number of RNN units
rnn_units = 1024 ## 1024

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

 # Be sure the vocabulary size matches the `StringLookup` layers. ######################################################## !!!!!!!!!!!
model = MyModel(vocab_size=len(ids_from_chars.get_vocabulary()), embedding_dim=embedding_dim, rnn_units=rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
  #print(type(input_example_batch), type(target_example_batch))
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


model.summary()
## sampled_indices, This gives us, at each timestep, a prediction of the next character index:
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

# Decode these to see the text predicted by this untrained model:
print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
print()
print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())

## Train the model

#At this point the problem can be treated as a standard classification problem. 
#Given the previous RNN state, and the input this time step, predict the class of the next character.

#Attach an optimizer, and a loss function
#The standard tf.keras.losses.sparse_categorical_crossentropy loss function works in this case because it is applied across the last dimension of the predictions.

#Because your model returns logits, you need to set the from_logits flag.

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", mean_loss)


tf.exp(mean_loss).numpy()

#Configure the training procedure using the tf.keras.Model.compile method. 
#Use tf.keras.optimizers.Adam with default arguments and the loss function.

model.compile(optimizer='adam', loss=loss)

# Configure checkpoints
## Use a tf.keras.callbacks.ModelCheckpoint to ensure that checkpoints are saved during training:

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

# Execute the training

EPOCHS = 50
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Each time you call the model you pass in some text and an internal state. 
# The model returns a prediction for the next character and its new state. 
# Pass the prediction and state back in to continue generating text.

#The following makes a single step prediction:

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]', '\n'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

## Run it in a loop to generate some text. 
## Looking at the generated text, you'll see the model knows when to capitalize, 
## make paragraphs and imitates a Shakespeare-like writing vocabulary.
## With the small number of training epochs, it has not yet learned to form coherent sentences.


start = time.time()
states = None
next_char = tf.constant(['the ', 'THE '])
result = [next_char]

for n in range(200):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)


## EXPORT THE GENERATOR

tf.saved_model.save(one_step_model, r"D:\Free_time_learning\DesktopApp\CalendarGUI\tral\dossier\model1")
one_step_reloaded = tf.saved_model.load(r"D:\Free_time_learning\DesktopApp\CalendarGUI\tral\dossier\model1")
states = None
next_char = tf.constant(['the ', 'THE '])
result = [next_char]

for n in range(200):
  next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
  result.append(next_char)


print(tf.strings.join(result)[0].numpy().decode("utf-8"))