import torch
import torch.utils.data as data
import librosa as lr
import scipy.io.wavfile as wf
import numpy as np
from torchvision import transforms

from collections import OrderedDict, defaultdict, Counter
from torch.utils.data.sampler import Sampler
import random
import math
import os
import functools
import time
import utils
import re
import mmh3

AUDIO_EXTENSIONS = ['.wav']

NUM_FOLDS = 10
SILENCE_LABEL = 'silence'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = 'unknown'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
KNOWN_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
UNKNOWN_LABELS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                  'bird', 'dog', 'cat', 'bed', 'house', 'tree', 'marvin', 'sheila', 'happy', 'wow']


def get_labels(train_unknown=True):
    if train_unknown:
        """
        Train the 'unknown' class by using all training set words
        that aren't in the known set
        """
        foreground_labels = KNOWN_LABELS
        background_labels = [SILENCE_LABEL, UNKNOWN_WORD_LABEL]
    else:
        foreground_labels = KNOWN_LABELS + UNKNOWN_LABELS
        background_labels = [SILENCE_LABEL]
    all_labels = background_labels + foreground_labels
    return all_labels


def find_commands(folder, types=AUDIO_EXTENSIONS):
    commands = []
    background = []
    for root, _, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path)
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filename = os.path.join(root, f)
                if label == BACKGROUND_NOISE_DIR_NAME:
                    background.append(filename)
                else:
                    parts = base.split('_nohash_')
                    if len(parts) == 2:
                        speaker_id = parts[0]
                        utter_id = int(parts[1])
                    else:
                        speaker_id = ''
                        utter_id = 0
                    commands.append((speaker_id, utter_id, label, filename))
    return commands, background


def speaker_to_fold(speaker, num_folds=NUM_FOLDS):
    hash = mmh3.hash(speaker, signed=False)
    p = 1 / float(2**32)
    j = 1 / num_folds
    fold = int(hash * p // j)
    return fold


class CommandsDataset(data.Dataset):
    def __init__(
            self,
            root='',
            mode='train',
            fold=0,
            wav_size=16000,
            format='raw',
            train_unknown=True,
            test_aug=0,
            transform=None):

        self.root = root
        self.mode = mode
        self.is_training = self.mode == 'train'
        self.fold = fold
        assert self.fold < NUM_FOLDS
        self.wav_size = wav_size
        self.format = format
        self.train_unknown = train_unknown

        # Training settings, TODO make dict, source from cmd args
        self.silence_prob = 0.11
        self.unknown_prob = 0.11 if train_unknown else 0.0
        self.background_frequency = 0.7
        self.background_volume_range = 0.25
        self.pitch_shift = 0.5
        self.pitch_shift_frequency = 0.0
        self.time_stretch = 0.15
        self.time_stretch_frequency = 0.5
        self.time_shift = 0.22

        # Find dataset input files
        commands, background = find_commands(root)
        print(len(commands), len(background))

        self.targets = []
        self.inputs = []
        self.class_inputs = OrderedDict()
        self.label_to_id = {}
        self.id_to_label = get_labels(train_unknown=train_unknown)
        for i, l in enumerate(self.id_to_label):
            self.label_to_id[l] = i
            self.class_inputs[i] = []
        self.background_data =[]
        if mode == 'test':
            for _, _, _, f in commands:
                self.inputs.append(f)
            self.inputs = np.array(self.inputs)
            self.targets = None
        else:
            self.unknowns = []
            self.num_known = 0
            index = 0
            for s, u, l, f in commands:
                sf = speaker_to_fold(s)
                if mode == 'validate':
                    if sf != fold:
                        continue
                else:
                    if sf == fold:
                        continue
                if l in self.label_to_id:
                    lid = self.label_to_id[l]
                    self.targets.append(lid)
                    self.inputs.append(f)
                    self.class_inputs[lid].append(index)
                    index += 1
                else:
                    assert train_unknown
                    self.unknowns.append(f)

            # handle unknown/silence sampling inline so no need for external sampler
            self.num_known = len(self.inputs)
            known_prob = 1.0 - (self.silence_prob + self.unknown_prob)
            dataset_size = int(self.num_known / known_prob)
            if train_unknown:
                unknown_size = int(dataset_size * self.unknown_prob)
            else:
                unknown_size = 0
            silence_size = int(dataset_size * self.silence_prob)
            if silence_size:
                self.inputs += [self.id_to_label[SILENCE_INDEX]] * silence_size
                self.targets += [SILENCE_INDEX] * silence_size
                self.class_inputs[SILENCE_INDEX] += list(range(index, index + silence_size))
                index += silence_size
            if unknown_size:
                if self.is_training:
                    self.inputs += [self.id_to_label[UNKNOWN_WORD_INDEX]] * unknown_size
                else:
                    self.inputs += list(np.random.choice(self.unknowns, unknown_size, replace=False))
                self.targets += [UNKNOWN_WORD_INDEX] * unknown_size
                self.class_inputs[UNKNOWN_WORD_INDEX] += list(range(index, index + unknown_size))
                index += unknown_size
            print(dataset_size, silence_size, unknown_size, known_prob)
            self.inputs = np.array(self.inputs)
            self.targets = np.array(self.targets)
            #shuffle_index = np.arange(self.inputs.shape[0])
            #np.random.shuffle(shuffle_index)
            #self.inputs = self.inputs[shuffle_index]
            #self.targets = self.targets[shuffle_index]
            for b in background:
                wav_audio, wav_rate = lr.load(b)
                self.background_data.append(wav_audio)

    def _process_sample(
            self,
            filename,
            target,
            pitch_shift=0.,
            pitch_shift_frequency=0.,
            time_stretch=0.,
            time_stretch_frequency=0.,
            time_shift=0.,
            ):
        # Data and labels will be populated and returned.
        desired_samples = 16000
        use_background = self.background_data and self.is_training

        # If we want silence, mute out the main sample but leave the background.
        if target == SILENCE_INDEX:
            foreground_volume = 0.
        else:
            foreground_volume = np.random.uniform(0.8, 1.0) if self.is_training else 1.0

        if self.train_unknown and (
                target == UNKNOWN_WORD_INDEX and filename == UNKNOWN_WORD_LABEL):
            filename = self.unknowns[np.random.randint(len(self.unknowns))]

        sample_rate = 16000
        if foreground_volume > 0:
            sample_rate, sample_audio = wf.read(filename)
            sample_audio = sample_audio.astype(np.float32) / 2**15
            #print(sample_audio.min(), sample_audio.max(), sample_audio.mean())

            if pitch_shift > 0 and np.random.uniform(0, 1) < pitch_shift_frequency:
                pitch_shift_amount = np.random.uniform(-pitch_shift, pitch_shift)
            else:
                pitch_shift_amount = 0
            #print('pitch shift: ', pitch_shift_amount)
            if pitch_shift_amount != 0:
                sample_audio = lr.effects.pitch_shift(sample_audio, sample_rate, pitch_shift_amount)
                time_stretch_amount = 1.0
            elif time_stretch > 0 and np.random.uniform(0, 1) < time_stretch_frequency:
                time_stretch_amount = np.random.uniform(1.0 - time_stretch, 1.0 + time_stretch)
            else:
                time_stretch_amount = 1.0
            #print('time stretch: ', time_stretch_amount)
            if time_stretch_amount != 1.0:
                sample_audio = lr.effects.time_stretch(sample_audio, time_stretch_amount)

            actual_samples = sample_audio.shape[0]
            # If we're time shifting, set up the offset for this sample.
            if time_shift > 0:
                time_shift_amount = int(desired_samples * time_shift)
                time_shift_amount = np.random.randint(-time_shift_amount, time_shift_amount)
            else:
                time_shift_amount = 0
            #print('time shift: ', time_shift_amount)
            if time_shift_amount < 0:
                crop_l = -time_shift_amount
                pad_l = 0
            else:
                crop_l = 0
                pad_l = time_shift_amount
            crop_r = min(actual_samples, desired_samples + crop_l - pad_l)
            pad_r = max(0, desired_samples - (crop_r - crop_l + pad_l))
            #print('cl, cr, pl, pr:', crop_l, crop_r, pad_l, pad_r)

            sample_audio = sample_audio[crop_l:crop_r]
            if pad_l and pad_r:
                sample_audio = np.r_[
                    np.random.uniform(-0.001, 0.001, pad_l).astype(np.float32),
                    sample_audio,
                    np.random.uniform(-0.001, 0.001, pad_r).astype(np.float32)]
            elif pad_l:
                sample_audio = np.r_[
                    np.random.uniform(-0.001, 0.001, pad_l).astype(np.float32),
                    sample_audio]
            elif pad_r:
                sample_audio = np.r_[
                    sample_audio,
                    np.random.uniform(-0.001, 0.001, pad_r).astype(np.float32)]
            if foreground_volume != 1.0:
                sample_audio *= foreground_volume
        else:
            sample_audio = np.zeros(desired_samples, dtype=np.float32)

        # Choose a section of background noise to mix in.
        if use_background and np.random.uniform(0, 1) < self.background_frequency:
            background_index = np.random.randint(len(self.background_data))
            background_samples = self.background_data[background_index]
            background_offset = np.random.randint(
                0, len(background_samples) - desired_samples)
            background_cropped = background_samples[
                background_offset:background_offset + desired_samples]
            if target == SILENCE_INDEX:
                background_volume = np.random.uniform(0.01, 0.08)
            else:
                background_volume = np.random.uniform(0.01, self.background_volume_range)
            sample_audio = (sample_audio + background_cropped * background_volume).clip(-1.0, 1.0)

        # lr.output.write_wav('./temp/lr-%d.wav' % i, sample_audio, 16000
        if self.format == 'spectrogram':
            spect = np.abs(lr.spectrum.stft(
                y=sample_audio, n_fft=512, win_length=480, hop_length=160, center=False))**2
            #print(spect.shape)
            mel = lr.feature.melspectrogram(S=spect, fmin=125, fmax=7500, n_mels=64)
            #print(mel.shape)
            log_mel = lr.spectrum.power_to_db(mel)
            mfcc = lr.feature.mfcc(S=log_mel, n_mfcc=64)
            mfcc_mean = np.mean(mfcc)
            mfcc_std = np.std(mfcc)
            mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-6)
            # mfcc = lr.feature.mfcc(
            #     y=sample_audio, sr=sample_rate,
            #     hop_length=192,
            #     n_fft=512,
            #     fmin=125,
            #     fmax=7500,
            #     power=2,
            #     n_mels=128,
            #     n_mfcc=64)
            #print(mfcc.shape)
            #mfcc = (mfcc - self.spect_mean) / self.spect_std
            return np.expand_dims(mfcc, axis=0).astype(np.float32)
        else:
            return sample_audio

    def __getitem__(self, index):
        filename = self.inputs[index]
        if self.targets is None:
            target = None
        else:
            target = self.targets[index]

        if self.is_training:
            sample = self._process_sample(
                filename, target,
                pitch_shift=self.pitch_shift,
                pitch_shift_frequency=self.pitch_shift_frequency,
                time_stretch=self.time_stretch,
                time_stretch_frequency=self.time_stretch_frequency,
                time_shift=self.time_shift)
        else:
            sample = self._process_sample(filename, target)

        if target is None:
            target = torch.zeros(1).long()
        return sample, target

    def __len__(self):
        return len(self.inputs)

    def filename(self, index, rel=True):
        abs_filename = self.inputs[index]
        if rel:
            return os.path.relpath(abs_filename, self.root)
        else:
            return abs_filename


class PKSampler(Sampler):

    def __init__(self, data_source, p=8, k=64):
        self.p = p
        self.k = k
        self.data_source = data_source

    def __iter__(self):
        pk_count = len(self) // (self.p * self.k)
        print(pk_count)
        for _ in range(pk_count):
            classes = torch.multinomial(
                torch.arange(len(self.data_source.class_inputs.keys())), self.p)
            for c in classes:
                inputs = self.data_source.class_inputs[c]
                for i in torch.randperm(len(inputs)).long()[:self.k]:
                    yield inputs[i]

    def __len__(self):
        pk = self.p * self.k
        return ((len(self.data_source) - 1) // pk + 1) * pk


def _test():
    ds = CommandsDataset(
        root='./commands/train/audio')
    print(ds.inputs)


if __name__ == '__main__':
    _test()
