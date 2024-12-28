import numpy as np
import torch
import math

# Do Pre-Emphasis
def pre_emphasis(signal, pre_emphasis_coeff=0.97):
    emphasized_signal = torch.cat((signal[:1], signal[1:] - pre_emphasis_coeff * signal[:-1]))
    return emphasized_signal

# Do Padding and Framing
def framing(signal, frame_size, frame_stride, sample_rate):
    frame_length = frame_size * sample_rate
    frame_step = frame_stride * sample_rate
    signal_length = signal.shape[0]
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(math.ceil(float(abs(signal_length - frame_length)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = torch.zeros(pad_signal_length - signal_length)
    pad_signal = torch.cat((signal, z))

    indices = torch.arange(0, frame_length).unsqueeze(0) + torch.arange(0, num_frames * frame_step, frame_step).unsqueeze(1)
    frames = pad_signal[indices.long()]
    return frames

# Add Hamming Window
def windowing(frames):
    window = torch.hamming_window(frames.shape[1])
    windowed_frames = frames * window
    return windowed_frames

# Do Fourier Transform
def compute_fft(frames, NFFT):
    complex_spectrum = torch.fft.rfft(frames, n=NFFT)
    power_spectrum = (complex_spectrum.real ** 2 + complex_spectrum.imag ** 2)
    return power_spectrum

# Do Mel Filter Banks
def mel_filter_bank(sample_rate, NFFT, nfilt, low_freq=0, high_freq=None):
    if high_freq is None:
        high_freq = sample_rate / 2

    def hz_to_mel(hz):
        return 2595 * torch.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    mel_low = hz_to_mel(torch.tensor(low_freq))
    mel_high = hz_to_mel(torch.tensor(high_freq))
    mel_points = torch.linspace(mel_low, mel_high, nfilt + 2)
    hz_points = mel_to_hz(mel_points)
    bin = torch.floor((NFFT + 1) * hz_points / sample_rate).long()

    fbank = torch.zeros(nfilt, NFFT//2 +1)
    for m in range(1, nfilt +1):
        f_m_minus = bin[m -1]
        f_m = bin[m]
        f_m_plus = bin[m +1]

        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k - bin[m-1]) / (bin[m] - bin[m-1] + 1e-8)
        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (bin[m +1] - k) / (bin[m +1] - bin[m] + 1e-8)
    return fbank

def apply_mel_filter(power_spectrum, fbank):
    mel_energies = torch.matmul(power_spectrum, fbank.t())
    mel_energies = torch.clamp(mel_energies, min=1e-10)
    return mel_energies

def get_f_bank_feats(wf, sample_rate=16000, pre_emphasis_coeff=0.97,
                              frame_size=0.025, frame_stride=0.01, NFFT=512, nfilt=40):
    emphasized_wf = pre_emphasis(wf, pre_emphasis_coeff)
    frames = framing(emphasized_wf, frame_size, frame_stride, sample_rate)
    windowed_frames = windowing(frames)
    power_spectrum = compute_fft(windowed_frames, NFFT)
    fbank = mel_filter_bank(sample_rate, NFFT, nfilt)
    mel_energies = apply_mel_filter(power_spectrum, fbank)
    log_mel_features = torch.log(mel_energies)
    return log_mel_features

# Get Silence Indices
def get_silence_indices(wf_np, sr):
    frame_length = 0.025
    frame_shift = 0.010
    frame_length_samples = int(frame_length * sr)
    frame_shift_samples = int(frame_shift * sr)
    num_frames = int((len(wf_np) - frame_length_samples) / frame_shift_samples) + 1
    frames = np.stack([
        wf_np[i * frame_shift_samples: i * frame_shift_samples + frame_length_samples]
        for i in range(num_frames)
    ])
    frame_energy = np.sum(frames ** 2, axis=1)

    energy_threshold = np.mean(frame_energy) * 0.5
    speech_frames = frame_energy > energy_threshold

    # Detect silence frames
    min_silence_duration = 1
    min_silence_frames = int(min_silence_duration / frame_shift)
    silence_indices = []
    current_silence = []

    for i, is_speech in enumerate(speech_frames):
        if not is_speech:
            current_silence.append(i)
        else:
            if len(current_silence) >= min_silence_frames:
                silence_start = current_silence[0] * frame_shift
                silence_end = (current_silence[-1] + 1) * frame_shift
                silence_indices.append((silence_start, silence_end))
            current_silence = []

    if len(current_silence) >= min_silence_frames:
        silence_start = current_silence[0] * frame_shift
        silence_end = (current_silence[-1] + 1) * frame_shift
        silence_indices.append((silence_start, silence_end))
    return silence_indices