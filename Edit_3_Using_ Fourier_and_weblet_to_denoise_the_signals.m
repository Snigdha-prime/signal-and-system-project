% Audio Noise Reduction for Voice Communication
% This script filters specific frequencies from an audio signal using wavelets and notch filters.

clear; clc; close all;

% Load the audio file
[filename, pathname] = uigetfile({'*.wav; *.mp3', 'Audio Files (*.wav, *.mp3)'}, 'Select an audio file');
if isequal(filename, 0) || isequal(pathname, 0)
    disp('User canceled the file selection.');
    return;
end
audioFilePath = fullfile(pathname, filename);
[audioIn, fs] = audioread(audioFilePath); % Read audio file

% Convert to mono if stereo
if size(audioIn, 2) > 1
    audioIn = mean(audioIn, 2); % Average channels to convert to mono
end

% Display original audio signal
figure; % New figure for original audio
plot((1:length(audioIn))/fs, audioIn, 'LineWidth', 1.5);
title('Original Audio Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
set(gca, 'FontSize', 12); % Increase font size for clarity
axis tight; % Adjust axes to fit the data

% Perform FFT on the original audio signal
N = length(audioIn);
audioFFT = fft(audioIn); % Compute FFT of audio signal
frequencies = (0:N-1) * (fs / N); % Frequency vector
audioFFTShifted = fftshift(audioFFT); % Shift zero frequency component to center

% Plot frequency domain of original audio signal
figure; % New figure for frequency domain of original audio
plot(frequencies - fs/2, abs(audioFFTShifted), 'LineWidth', 1.5);
title('Frequency Domain of Original Audio Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([-fs/2 fs/2]);
grid on;
set(gca, 'FontSize', 12); % Increase font size for clarity

% Wavelet Denoising
waveletName = 'db8'; % Daubechies wavelet
[coeffs, levels] = wavedec(audioIn, 5, waveletName); % Decompose audio signal

% Universal threshold calculation
sigma = median(abs(coeffs)) / 0.6745; % Estimate noise standard deviation
threshold = sigma * sqrt(2 * log(length(audioIn))); % Universal threshold

% Apply thresholding
coeffsDenoised = coeffs; % Initialize denoised coefficients
coeffsDenoised(abs(coeffs) < threshold) = 0; % Apply thresholding

% Reconstruct the denoised signal
audioFiltered = waverec(coeffsDenoised, levels, waveletName);

% Display filtered audio signal after wavelet denoising
figure; % New figure for filtered audio
plot((1:length(audioFiltered))/fs, audioFiltered, 'LineWidth', 1.5);
title('Filtered Audio Signal (Wavelet Denoising)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
set(gca, 'FontSize', 12); % Increase font size for clarity
axis tight; % Adjust axes to fit the data

% Apply notch filtering to remove specific noise frequencies
notchFreqs = [60, 120, 240]; % Example noise frequencies in Hz (e.g., 60 Hz hum)
bw = 2; % Bandwidth for the notch filters

for f = notchFreqs
    % Design a notch filter for each frequency
    if f > 0 % Ensure positive frequency
        notchFreq = [f - bw/2, f + bw/2] / (fs / 2); % Normalize frequency
        [bNotch, aNotch] = butter(2, notchFreq, 'stop'); % 2nd order notch filter
        audioFiltered = filtfilt(bNotch, aNotch, audioFiltered); % Apply notch filter
    end
end

% Apply a bandpass filter to enhance the human voice frequencies
lowCutoff = 300;   % Lower cutoff frequency (Hz)
highCutoff = 3400; % Upper cutoff frequency (Hz)

% Normalize cutoff frequencies for the Butterworth filter
nyquist = fs / 2;
lowCutoffNorm = lowCutoff / nyquist;
highCutoffNorm = highCutoff / nyquist;

% Design a bandpass filter to isolate human voice frequencies
[b, a] = butter(4, [lowCutoffNorm highCutoffNorm], 'bandpass'); % 4th order Butterworth filter

% Apply the filter using filtfilt to avoid phase distortion
audioFiltered = filtfilt(b, a, audioFiltered);

% Amplify the filtered audio slightly for clearer speech
audioFiltered = audioFiltered * 1.5; % Increase the gain factor for amplification

% Normalize to prevent clipping
audioFiltered = audioFiltered / max(abs(audioFiltered)); % Normalize the amplified audio

% Display final filtered audio signal
figure; % New figure for final filtered audio
plot((1:length(audioFiltered))/fs, audioFiltered, 'LineWidth', 1.5);
title('Final Filtered Audio Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
set(gca, 'FontSize', 12); % Increase font size for clarity
axis tight; % Adjust axes to fit the data

% Perform FFT on the final filtered audio signal
audioFinalFFT = fft(audioFiltered);
audioFinalFFTShifted = fftshift(audioFinalFFT); % Shift zero frequency component to center

% Plot frequency domain of final filtered audio signal
figure; % New figure for frequency domain of filtered audio
plot(frequencies - fs/2, abs(audioFinalFFTShifted), 'LineWidth', 1.5);
title('Frequency Domain of Final Filtered Audio Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([-fs/2 fs/2]);
grid on;
set(gca, 'FontSize', 12); % Increase font size for clarity

% Save the final filtered audio to a new WAV file
outputFilePath = fullfile(pathname, 'filtered_audio.wav');
audiowrite(outputFilePath, audioFiltered, fs); % Save filtered audio
disp(['Filtered audio saved to: ', outputFilePath]);