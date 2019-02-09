#
# EPITECH PROJECT, 2019
# LapiscineIaCestLeFeu
# File description:
# Mozart-like AI
#

import tensorflow as tf
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
import os
import tensorflow.keras as keras
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LEARNING_RATE = 0.1
EPOCHS = 10

class Train_Neural_Network:
    """" Will serve to train our AI """
    def __init__(self):
        self.notes = self.get_notes()
        self.n_notes = len(set(self.notes))
        self.input, self.output = self.prepare_in_and_out(self.notes, self.n_notes)
        self.model = get_model()

    def get_notes(self):
        """ Will get all the notes from all the midi files """
        notes = []
        for music in glob.glob("dataset/*.mid"):
            print("Getting notes from {}".format(music))

            fd = converter.parse(music)
            instruments = instrument.partitionByInstrument(fd)
            if instruments:
                all_notes = instruments.parts[0].recurse()
            else:
                all_notes = fd.flat.notes

            for machin in all_notes:
                if isinstance(machin, note.Note):
                    notes.append(str(machin.pitch))
                elif isinstance(machin, chord.Chord):
                    notes.append('.'.join(str(n) for n in machin.normalOrder))
        return notes

    def prepare_in_and_out(self, notes, n_notes):
        """ Convert extracted midi data to a proper format """
        seq_len = 20
        notes_names = sorted(set(n for n in notes))
        note_to_int = dict((note, number) for number, note in enumerate(notes_names))
        input = []
        output = []
        for i in range(len(notes) - seq_len):
            seq_in = notes[i:i + seq_len]
            seq_out = notes[i + seq_len]
            input.append([note_to_int[note] for note in seq_in])
            output.append(note_to_int[seq_out])
        n_seq = len(input)
        input = np.reshape(input, (n_seq, seq_len, 1))
        input = input / float(n_notes)
        output = keras.utils.to_categorical(output)
        return input, output

    def save_model(self):
        """ Save the model created before (not so useful) """
        model.save('model_save.h5')
        del model

    def load_model(self):
        """ Load the model saved before (not so useful) """
        return tf.keras.load_model('model_save.h5')

    def get_model(self):
        """ Create the model that we will use """
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.LSTM(
        512, input_shape=(self.input.shape[1], self.input.shape[2]), return_sequences=True
        ))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.LSTM(512, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.LSTM(512))
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(self.n_notes))
        model.add(tf.keras.layers.Activation('softmax'))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, model, input, output):
        """ Train the AI with given model, inputs and outputs, and make checkpoints at each epoch"""
        filepath = "weights-at-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss',
                                               verbose=0,
                                               save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        model.fit(input, output, epochs=100, batch_size=4, callbacks=callbacks_list)

class Generate_Neural_Network:
    def __init__(self, input):
        self.notes = self.get_notes()
        self.n_notes = len(set(notes))
        self.input, self.output = prepare_in_and_out(notes, n_notes)
        self.model = get_model()



if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 2 or argv[1] == "-h" or argv[1] == "--help":
        print_help()
        exit(1)
    if argv[1] == "-t" or argv[1] == "--train":
        nn = Train_Neural_Network()
        nn.train(nn.model, nn.input, nn.output)
        exit(0)
    elif argv[1] == "-r" or argv[1] == "--run":
        nn = Run_NeuralNetwork()
        nn.run_run()
        exit(0)
