import audiomentations
import numpy as np
import torch
import torchaudio
import torchvision
import colorednoise as cn


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class Bird2023Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, X, y=None, train=True):
        self.cfg = cfg
        self.train = train
        self.df = X
        self.audio_length = cfg["audio"]["sample_rate"] * self.cfg["audio"]["train_duration"] if train else cfg["audio"]["sample_rate"] * self.cfg["audio"]["valid_duration"] 
        
        # If labels are not provided, create a zero tensor of shape (len(X), num_columns_X)
        if y is None:
            self.y = torch.zeros(
                (len(self.df), len(self.df.columns)), dtype=torch.float32
            )
        # Otherwise, convert the labels to a tensor of dtype float32
        else:
            self.y = torch.tensor(y.values, dtype=torch.float32)

        # Define augmentation methods for the waveform
        self.augmentation_backgroundnoise = audiomentations.OneOf(
                    [
                        audiomentations.AddBackgroundNoise(
                            sounds_path=f"{cfg['general']['input_path']}/ff1010bird_nocall/nocall",
                            min_snr_in_db=0,
                            max_snr_in_db=3,
                            p=0.5,
                        ),
                        audiomentations.AddBackgroundNoise(
                            sounds_path=f"{cfg['general']['input_path']}/train_soundscapes/nocall",
                            min_snr_in_db=0,
                            max_snr_in_db=3,
                            p=0.25,
                        ),
                        audiomentations.AddBackgroundNoise(
                            sounds_path=f"{cfg['general']['input_path']}/aicrowd2020_noise_30sec/noise_30sec",
                            min_snr_in_db=0,
                            max_snr_in_db=3,
                            p=0.25,
                        ),
                    ],
                    p=0.5,
                )
        self.augmentation_gaussiannoise = audiomentations.OneOf(
                    [
                        audiomentations.AddGaussianSNR(p=0.5),
                        audiomentations.AddGaussianNoise(p=0.5)
                    ],
                    p=0.5
                )
        self.augmentation_gain = audiomentations.OneOf(
                    [
                        audiomentations.Gain(p=0.5),
                        audiomentations.GainTransition(p=0.5),
                    ],
                    p=0.5
                )
        self.augmentation_pinknoise = PinkNoise(p=0.5)

        # Define a normalization transformation for the waveform
        self.normalize_waveform = audiomentations.Normalize(p=1.0)

        # Define augmentation methods for the me_specgram
        self.aug_timestretch = torchaudio.transforms.TimeStretch()

        # Define a normalization transformation for the mel_specgram
        self.normalize_melspecgram = torchvision.transforms.Normalize(
            mean=self.cfg["model"]["mean"], std=self.cfg["model"]["std"]
        )

    def __len__(self):
        return len(self.df)

    def min_max_0_1(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    def z_normalize(self, x):
        mean = torch.mean(x)
        std = torch.std(x) + 1e-6
        x_normalized = (x - mean) / std
        return x_normalized

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
        # Retrieve the label of the audio file at the given index
        y = self.y[index]
        # Load the audio waveform and its sample rate from the file path
        if self.cfg["job_type"] == "pretrain":
            # Retrieve the filename of the audio file at the given index
            file_path = self.df.loc[index, "filepath"]
            waveform, sample_rate = torchaudio.load(
                self.cfg["general"]["pretrain_input_path"] + "/" + file_path + ".ogg"
            )
        else:
            # Retrieve the filename of the audio file at the given index
            file_path = self.df.loc[index, "filename"]
            waveform, sample_rate = torchaudio.load(
                self.cfg["general"]["input_path"] + "/" + file_path
            )

        # Resample the waveform if the sample rate is not equal to the target sample rate
        if sample_rate != self.cfg["audio"]["sample_rate"]:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.cfg["audio"]["sample_rate"]
            )

        # Repeat waveform
        waveform = self.repeat_waveform(waveform, self.audio_length)
        # Crop or pad the waveform to a fixed duration
        if self.train and torch.rand(1) >= 0.5:
            waveform = self.crop_or_pad_waveform_random(waveform, self.audio_length)
        else:
            waveform = self.crop_or_pad_waveform_constant(waveform, self.audio_length)
        
        # Stereo to mono
        if waveform.shape[0] == 2:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # To numpy
        waveform = waveform.numpy()
        waveform = np.squeeze(waveform)
        if self.train:
            # Apply waveform augmentations
            waveform = self.augmentation_backgroundnoise(
                samples=waveform, sample_rate=self.cfg["audio"]["sample_rate"]
            )
            waveform = self.augmentation_gaussiannoise(
                samples=waveform, sample_rate=self.cfg["audio"]["sample_rate"]
            )
            #waveform = self.augmentation_gain(samples=waveform, sample_rate=self.cfg["audio"]["sample_rate"])
        # Apply normalization
        waveform = self.normalize_waveform(
            samples=waveform, sample_rate=self.cfg["audio"]["sample_rate"]
        )
        # To tensor
        waveform = waveform[np.newaxis, :]
        waveform = torch.from_numpy(waveform)

        # Compute the mel spectrogram of the waveform
        mel_specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg["audio"]["sample_rate"],
            n_fft=self.cfg["mel_specgram"]["n_fft"],
            win_length=self.cfg["mel_specgram"]["win_length"],
            hop_length=self.cfg["mel_specgram"]["hop_length"],
            f_min=self.cfg["mel_specgram"]["f_min"],
            f_max=self.cfg["mel_specgram"]["f_max"],
            n_mels=self.cfg["mel_specgram"]["n_mels"],
        )(waveform)
        # Convert the mel spectrogram to a decibel scale
        mel_specgram = torchaudio.transforms.AmplitudeToDB(
            top_db=self.cfg["mel_specgram"]["top_db"]
        )(mel_specgram)

        # Apply mel spectrogram augmentations
        if self.train and torch.rand(1) >= 0.5:
            mel_specgram = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=mel_specgram.shape[1] // 5
            )(mel_specgram)
            mel_specgram = torchaudio.transforms.TimeMasking(
                time_mask_param=mel_specgram.shape[2] // 5
            )(mel_specgram)
            # mel_specgram = self.aug_timestretch(mel_specgram)
            
        # Apply z-norm, each melspec
        #mel_specgram = self.z_normalize(mel_specgram)

        # Apply min-max normalization to scale values between 0 and 1
        mel_specgram = self.min_max_0_1(mel_specgram)

        # Expand to n channels
        if self.cfg["model"]["in_chans"] > 1:
            mel_specgram = mel_specgram.repeat(self.cfg["model"]["in_chans"], 1, 1)

        # Apply z-normalization to scale values to have zero mean and unit variance
        mel_specgram = self.normalize_melspecgram(mel_specgram)
        
        mel_specgram = torch.nan_to_num(mel_specgram)

        return mel_specgram, y


class Bird2023TestDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, ogg_name_list):
        self.cfg = cfg
        self.ogg_name_list = ogg_name_list
        self.audio_length = cfg["audio"]["sample_rate"] * self.cfg["audio"]["test_duration"]
        self.step = cfg["audio"]["sample_rate"] * 5

        # Define a normalization transformation for the waveform
        self.normalize_waveform = audiomentations.Normalize(p=1.0)

        # Define a normalization transformation for the mel_specgram
        self.normalize_melspecgram = torchvision.transforms.Normalize(
            mean=self.cfg["model"]["mean"], std=self.cfg["model"]["std"]
        )

    def __len__(self):
        return len(self.ogg_name_list)

    def min_max_0_1(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def audio_to_mel_specgram(self, waveform):
        # To numpy
        waveform = waveform.numpy()
        waveform = np.squeeze(waveform)
        # Apply normalization
        waveform = self.normalize_waveform(
            samples=waveform, sample_rate=self.cfg["audio"]["sample_rate"]
        )
        # To tensor
        waveform = waveform[np.newaxis, :]
        waveform = torch.from_numpy(waveform)

        # Compute the mel spectrogram of the waveform
        mel_specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg["audio"]["sample_rate"],
            n_fft=self.cfg["mel_specgram"]["n_fft"],
            win_length=self.cfg["mel_specgram"]["win_length"],
            hop_length=self.cfg["mel_specgram"]["hop_length"],
            f_min=self.cfg["mel_specgram"]["f_min"],
            f_max=self.cfg["mel_specgram"]["f_max"],
            n_mels=self.cfg["mel_specgram"]["n_mels"],
        )(waveform)
        # Convert the mel spectrogram to a decibel scale
        mel_specgram = torchaudio.transforms.AmplitudeToDB(
            top_db=self.cfg["mel_specgram"]["top_db"]
        )(mel_specgram)
        
        # Apply z-norm, each melspec
        #mel_specgram = self.z_normalize(mel_specgram)

        # Apply min-max normalization to scale values between 0 and 1
        mel_specgram = self.min_max_0_1(mel_specgram)

        # Expand to n channels
        if self.cfg["model"]["in_chans"] > 1:
            mel_specgram = mel_specgram.repeat(self.cfg["model"]["in_chans"], 1, 1)

        # Apply z-normalization to scale values to have zero mean and unit variance
        mel_specgram = self.normalize_melspecgram(mel_specgram)
        mel_specgram = torch.nan_to_num(mel_specgram)

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
