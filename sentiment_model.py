import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflowjs as tfjs
import os

class SentimentModel:

    def __init__(self, path, BUFFER_SIZE = 10000, BATCH_SIZE = 64):

        """ Initialize matplotlib helper function 
        and dataset """

        self.path = path
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.save = save
        self.load_dataset()
        self.create_model()


    def plot_graphs(self):

        """ Create helper function to plot data after 
        training """

        plt.plot(self.history.history[self.metric])
        plt.plot(self.history.history['val_' + self.metric], '')
        plt.xlabel("Epochs")
        plt.ylabel(self.metric)
        plt.legend([self.metric, 'val_'+self.metric])
        plt.show()

    def load_dataset(self):

        """ Allocate data to training and testing from dataset """
        
        dataset, info = tfds.load(self.path, with_info=True, as_supervised=True)

        self.train_dataset, self.test_dataset = dataset['train'], dataset['test']

        self.encoder = info.features['text'].encoder
        self.vocab_size = self.encoder.vocab_size

        print ('Vocabulary size: {}'.format(self.vocab_size))

        self.train_dataset = self.train_dataset.shuffle(self.BUFFER_SIZE)
        self.train_dataset = self.train_dataset.padded_batch(self.BATCH_SIZE)

        self.test_dataset = self.test_dataset.padded_batch(self.BATCH_SIZE)

    def create_model(self):

        def sample_predict(sample_pred_text, pad):

            """ Mask the padding applied to the sequences """ 
            encoded_sample_pred_text = self.encoder.encode(sample_pred_text)

            if pad:
                encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
                encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
                predictions = self.model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

            return (predictions)
        
        def pad_to_size(vec, size):
            zeros = [0] * (size - len(vec))
            vec.extend(zeros)
            return vec

        """ Create layers for Recurrent Neural Network and uses an
        embedding layers to sequence words into vectors that with training 
        have similar values to those words with similar meanings """

        self.model = tf.keras.Sequential([
        tf.keras.layers.Embedding(self.encoder.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
        ])

        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

        self.history = self.model.fit(self.train_dataset, epochs=10,
                    validation_data=self.test_dataset,
                    validation_steps=30)

        self.test_loss, self.test_acc = self.model.evaluate(self.test_dataset)

        print('Test Loss: {}'.format(self.test_loss))
        print('Test Accuracy: {}'.format(self.test_acc))

        sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
        predictions = sample_predict(sample_pred_text, pad=True)
        print(predictions)
        
        
    

        
    def test_encodings(self, sample_string):
        
        """ Testing playground for asserting that our encoder works """

        encoded_string = self.encoder.encode(sample_string)
        print(' Encoded string is {}'.format(sample_string))

        original_string = self.encoder.decode(encoded_string)
        print(' Original string is {}'.format(original_string))

        assert original_string == sample_string

        for index in encoded_string:
            print('{} ----> {}'.format(index, self.encoder.decode([index])))

    def save_model(self, path, options="js"):

        if options == "js": 
            tfjs.converters.save_keras_model(self.model, path)
        elif options == "py":
            self.model.save("my_model")




if __name__ == '__main__':
    sentimentModel = SentimentModel('imdb_reviews/subwords8k')
    sentimentModel.test_encodings('Hello Tensorflow.')
