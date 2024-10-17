import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
import soundfile as sf

def load_wav(file_path, sample_rate=16000):
    audio, sr = sf.read(file_path)
    if sr != sample_rate:
        # You might want to implement resampling here
        raise ValueError(f"Sample rate mismatch: expected {sample_rate}, got {sr}")
    return Tensor(audio.astype(np.float32))

def preemphasis(signal, coeff=0.97):
    return Tensor.cat([signal[:1], signal[1:] - coeff * signal[:-1]])

def frame(signal, frame_length, frame_step):
    signal_length = signal.shape[0]
    frame_length = int(frame_length)
    frame_step = int(frame_step)
    num_frames = 1 + (signal_length - frame_length) // frame_step
    indices = Tensor.arange(0, frame_length).unsqueeze(0) + \
              Tensor.arange(0, num_frames * frame_step, frame_step).unsqueeze(1)
    return signal[indices]

def periodic_hann(window_length):
    return 0.5 - 0.5 * Tensor.cos(2 * np.pi * Tensor.arange(window_length) / window_length)

def stft(frames, fft_length):
    window = periodic_hann(frames.shape[1])
    windowed_frames = frames * window
    return Tensor.fft(Tensor.pad(windowed_frames, ((0, 0), (0, fft_length - frames.shape[1]))), 1)

def mel_filterbank(num_mel_bins, num_fft_bins, sample_rate, lower_edge_hertz=80.0, upper_edge_hertz=7600.0):
    nyquist_hertz = sample_rate / 2.0
    linear_freqs = Tensor.linspace(0.0, nyquist_hertz, num_fft_bins)
    mel_low = 2595.0 * Tensor.log10(1.0 + lower_edge_hertz / 700.0)
    mel_high = 2595.0 * Tensor.log10(1.0 + upper_edge_hertz / 700.0)
    mel_points = Tensor.linspace(mel_low, mel_high, num_mel_bins + 2)
    hz_points = 700.0 * (Tensor.pow(10.0, mel_points / 2595.0) - 1.0)
    bin = Tensor.floor((num_fft_bins + 1) * hz_points / sample_rate).astype(dtypes.int32)

    fbank = Tensor.zeros((num_mel_bins, num_fft_bins))
    for m in range(1, num_mel_bins + 1):
        f_m_minus = bin[m - 1]
        f_m = bin[m]
        f_m_plus = bin[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fbank

def dct(x, n_mfcc=13):
    N = x.shape[1]
    n = Tensor.arange(N)
    k = Tensor.arange(n_mfcc).unsqueeze(1)
    dct_matrix = Tensor.cos(np.pi / N * (n + 0.5) * k)
    return x @ dct_matrix

def extract_mfccs(signal, sample_rate=16000, frame_length=25, frame_step=10, num_mel_bins=40, num_mfccs=13):
    frame_length = int(frame_length * sample_rate / 1000)
    frame_step = int(frame_step * sample_rate / 1000)
    
    emphasized_signal = preemphasis(signal)
    frames = frame(emphasized_signal, frame_length, frame_step)
    
    fft_length = 2
    while fft_length < frame_length:
        fft_length *= 2

    mag_frames = stft(frames, fft_length).abs()
    
    mel_fbank = mel_filterbank(num_mel_bins, fft_length // 2 + 1, sample_rate)
    mel_spec = (mag_frames @ mel_fbank.T).log()
    
    mfccs = dct(mel_spec, n_mfcc=num_mfccs)
    
    return mfccs

# Example usage
if __name__ == "__main__":
    wav_file = "path/to/your/audio.wav"
    audio = load_wav(wav_file)
    mfccs = extract_mfccs(audio)
    print(f"MFCCs shape: {mfccs.shape}")
