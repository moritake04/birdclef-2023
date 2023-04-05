import audiomentations
import torch
import torchaudio
import torchvision


class Bird2023Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, X, y=None, augmentation=False):
        self.cfg = cfg
        self.augmentation = augmentation
        self.df = X
        self.audio_length = cfg["audio"]["sample_rate"] * self.cfg["audio"]["duration"]

        # If labels are not provided, create a zero tensor of shape (len(X), num_columns_X)
        if y is None:
            self.y = torch.zeros(
                (len(self.df), len(self.df.columns)), dtype=torch.float32
            )
        # Otherwise, convert the labels to a tensor of dtype float32
        else:
            self.y = torch.tensor(y.values, dtype=torch.float32)

        # Define augmentation methods for the waveform
        self.augmentation_waveform = audiomentations.Compose(
            [
                audiomentations.OneOf(
                    [
                        audiomentations.Gain(p=0.5),
                        audiomentations.GainTransition(p=0.5),
                    ]
                ),
                audiomentations.AddGaussianSNR(p=0.5),
                # audiomentations.AddBackgroundNoise(
                #    sounds_path=self.config.BACKGROUND_PATH,
                #    min_snr_in_db=0,
                #    max_snr_in_db=2,
                #    p=0.5,
                # ),
            ]
        )

        # Define a normalization transformation for the waveform
        self.normalize_waveform = audiomentations.Normalize(p=1.0)

        # Define augmentation methods for the waveform

        # Define a normalization transformation for the mel_specgram
        self.normalize_melspecgram = torchvision.transforms.Normalize(
            mean=self.cfg["model"]["mean"], std=self.cfg["model"]["std"]
        )

    def __len__(self):
        return len(self.df)

    def min_max_0_1(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def repeat_waveform(self, audio, target_len):
        # Get the length of the audio waveform
        audio_len = audio.shape[1]
        # Calculate the number of times the audio needs to be repeated
        repeat_num = (target_len // audio_len) + 1
        # Repeat the audio
        audio = audio.repeat(1, repeat_num)
        return audio

    def crop_or_pad_waveform_random(self, audio, target_len):
        # Get the length of the audio waveform
        audio_len = audio.shape[1]
        # If the length of the input audio is smaller than the target length, randomly pad the audio
        if audio_len < target_len:
            # Calculate the offset between the input audio and the target length
            diff_len = target_len - audio_len
            # Select a random location for padding
            pad_left = torch.randint(0, diff_len, size=(1,))
            pad_right = diff_len - pad_left
            # Apply padding to the audio data
            audio = torch.nn.functional.pad(
                audio, (pad_left, pad_right), mode="constant", value=0
            )
        # If the length of the input audio is larger than the target length, crop the audio
        elif audio_len > target_len:
            # Select a random location for cropping
            idx = torch.randint(0, audio_len - target_len, size=(1,))
            # Crop the audio data
            audio = audio[:, idx : (idx + target_len)]
        # Return the cropped or padded audio data
        return audio

    def crop_or_pad_waveform_constant(self, audio, target_len):
        # Get the length of the audio waveform
        audio_len = audio.shape[1]
        # If the length of the input audio is smaller than the target length, pad the audio
        if audio_len < target_len:
            # Calculate the offset between the input audio and the target length
            diff_len = target_len - audio_len
            # Pad the audio data from the left with zeros
            audio = torch.nn.functional.pad(
                audio, (0, diff_len), mode="constant", value=0
            )
        # If the length of the input audio is larger than the target length, crop the audio
        elif audio_len > target_len:
            # Crop the audio data from the beginning to target_len
            audio = audio[:, :target_len]
        # Return the cropped or padded audio data
        return audio

    def __getitem__(self, index):
        # Retrieve the filename of the audio file at the given index
        file_path = self.df.loc[index, "filename"]
        # Retrieve the label of the audio file at the given index
        y = self.y[index]
        # Load the audio waveform and its sample rate from the file path
        waveform, sample_rate = torchaudio.load(
            self.cfg["general"]["input_path"] + "/train_audio/" + file_path
        )

        # Resample the waveform if the sample rate is not equal to the target sample rate
        if sample_rate != self.cfg["audio"]["sample_rate"]:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.cfg["audio"]["sample_rate"]
            )

        # Repeat waveform
        waveform = self.repeat_waveform(waveform, self.audio_length)
        # Crop or pad the waveform to a fixed duration
        waveform = self.crop_or_pad_waveform_constant(waveform, self.audio_length)

        # To numpy
        waveform = waveform.numpy()
        # Apply waveform augmentations
        waveform = self.augmentation_waveform(
            samples=waveform, sample_rate=self.cfg["audio"]["sample_rate"]
        )
        # Apply normalization
        waveform = self.normalize_waveform(
            samples=waveform, sample_rate=self.cfg["audio"]["sample_rate"]
        )
        # To tensor
        waveform = torch.from_numpy(waveform)

        # Compute the mel spectrogram of the waveform
        mel_specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg["audio"]["sample_rate"],
            n_fft=self.cfg["mel_specgram"]["n_fft"],
            win_length=self.cfg["mel_specgram"]["win_length"],
            hop_length=self.cfg["mel_specgram"]["hop_length"],
            n_mels=self.cfg["mel_specgram"]["n_mels"],
        )(waveform)
        # Convert the mel spectrogram to a decibel scale
        mel_specgram = torchaudio.transforms.AmplitudeToDB()(mel_specgram)

        # Apply mel spectrogram augmentations

        # Expand to n channels
        if self.cfg["model"]["in_chans"] > 1:
            mel_specgram = mel_specgram.repeat(self.cfg["model"]["in_chans"], 1, 1)

        # Apply min-max normalization to scale values between 0 and 1
        mel_specgram = self.min_max_0_1(mel_specgram)
        # Apply z-normalization to scale values to have zero mean and unit variance
        mel_specgram = self.normalize_melspecgram(mel_specgram)

        return mel_specgram, y


class Bird2023TestDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, ogg_name_list):
        self.cfg = cfg
        self.ogg_name_list = ogg_name_list
        self.audio_length = cfg["audio"]["sample_rate"] * self.cfg["audio"]["duration"]
        self.step = cfg["audio"]["sample_rate"] * 5

        # Define a normalization transformation for the data
        self.normalize = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=self.cfg["model"]["mean"], std=self.cfg["model"]["std"]
                )
            ]
        )

    def __len__(self):
        return len(self.ogg_name_list)

    def min_max_0_1(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def audio_to_mel_specgram(self, audio):
        # Compute the mel spectrogram of the waveform
        mel_specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg["audio"]["sample_rate"],
            n_fft=self.cfg["mel_specgram"]["n_fft"],
            win_length=self.cfg["mel_specgram"]["win_length"],
            hop_length=self.cfg["mel_specgram"]["hop_length"],
            f_min=self.cfg["mel_specgram"]["f_min"],
            f_max=self.cfg["mel_specgram"]["f_max"],
            n_mels=self.cfg["mel_specgram"]["n_mels"],
        )(audio)
        # Convert the mel spectrogram to a decibel scale
        mel_specgram = torchaudio.transforms.AmplitudeToDB()(mel_specgram)

        # Expand to n channels
        if self.cfg["model"]["in_chans"] > 1:
            mel_specgram = mel_specgram.repeat(self.cfg["model"]["in_chans"], 1, 1)

        # Apply min-max normalization to scale values between 0 and 1
        mel_specgram = self.min_max_0_1(mel_specgram)
        # Apply z-normalization to scale values to have zero mean and unit variance
        mel_specgram = self.normalize(mel_specgram)

        return mel_specgram

    def __getitem__(self, index):
        # Retrieve the filename of the audio file at the given index
        file_path = self.ogg_name_list[index]
        # Load the audio waveform and its sample rate from the file path
        waveform, sample_rate = torchaudio.load(
            self.cfg["general"]["input_path"] + "/test_soundscapes/" + file_path
        )

        # Resample the waveform if the sample rate is not equal to the target sample rate
        if sample_rate != self.cfg["audio"]["sample_rate"]:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.cfg["audio"]["sample_rate"]
            )

        waveforms = []
        row_id = []
        for i in range(self.audio_length, waveform.shape[1] + self.step, self.step):
            start = max(0, i - self.audio_length)
            end = start + self.audio_length
            waveforms.append(waveform[:, start:end])
            row_id.append(f"{file_path[:-4]}_{end//self.cfg['audio']['sample_rate']}")

        if waveforms[-1].shape[1] < self.audio_length:
            waveforms = waveforms[:, :-1]
            row_id = row_id[:-1]

        mel_specgrams = [self.audio_to_mel_specgram(waveform) for waveform in waveforms]
        mel_specgrams = torch.stack(mel_specgrams)

        return mel_specgrams
