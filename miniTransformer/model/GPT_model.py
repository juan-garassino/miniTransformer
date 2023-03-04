import tensorflow as tf
from miniTransformer.model.transformer_block import TransformerBlock
import numpy as np

# Define the GPT model architecture
class GPTModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_size, num_layers, num_heads, dff,
                 dropout_rate, max_len):
        super(GPTModel, self).__init__()

        # Define the embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)

        # Define the positional encoding layer
        self.pos_encoding = self.positional_encoding(max_len, embedding_size)

        # Define the transformer layers
        self.transformer_layers = [
            TransformerBlock(num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        self.num_layers = num_layers
        # Define the output layer
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        # Embed the input tokens
        x = self.embedding(inputs)

        # Add the positional encoding
        x += self.pos_encoding[:, :tf.shape(x)[1], :]

        # Pass the input through the transformer layers
        for i in range(self.num_layers):
            x = self.transformer_layers[i](x)

        # Pass the output through the output layer
        output = self.output_layer(x)

        return output

    def positional_encoding(self, max_len, embedding_size):
        # Define the position indices
        position = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, embedding_size, 2, dtype=tf.float32) *
            -(np.log(10000.0) / embedding_size))

        # Compute the positional encoding values
        sin = tf.math.sin(position * div_term)
        cos = tf.math.cos(position * div_term)
        pos_encoding = tf.concat([sin, cos], axis=-1)[tf.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)
