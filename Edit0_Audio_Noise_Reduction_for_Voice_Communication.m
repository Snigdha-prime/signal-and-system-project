% Audio Noise Reduction for Voice Communication
% This script filters specific frequencies from an audio signal.

clear; clc; close all;

% Load the audio file
[filename, pathname] = uigetfile('*.wav', 'Select a WAV audio file');
if isequal(filename, 0) || isequal(pathname, 0)
    disp('User  canceled the file selection.');
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
plot((1:length(audioIn))/fs, audioIn);
title('Original Audio Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Perform FFT on the original audio signal
N = length(audioIn);
audioFFT = fft(audioIn); % Compute FFT of audio signal
frequencies = (0:N-1) * (fs / N); % Frequency vector
audioFFTShifted = fftshift(audioFFT); % Shift zero frequency component to center

% Plot frequency domain of original audio signal
figure; % New figure for frequency domain of original audio
plot(frequencies - fs/2, abs(audioFFTShifted));
title('Frequency Domain of Original Audio Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

% Define human voice frequency range
lowCutoff = 300;   % Lower cutoff frequency (Hz)
highCutoff = 3000; % Upper cutoff frequency (Hz)

% Normalize cutoff frequencies for the Butterworth filter
nyquist = fs / 2;
lowCutoffNorm = lowCutoff / nyquist;
highCutoffNorm = highCutoff / nyquist;

% Ensure cutoff frequencies are within the valid range
if lowCutoffNorm <= 0 || highCutoffNorm >= 1 || lowCutoffNorm >= highCutoffNorm
    error('Cutoff frequencies must be within the interval (0, 1) and lowCutoff < highCutoff.');
end

% Design a bandpass filter to isolate human voice frequencies
[b, a] = butter(4, [lowCutoffNorm highCutoffNorm], 'bandpass'); % 4th order Butterworth filter

% Apply the filter to the audio signal
audioFiltered = filter(b, a, audioIn);

% Display filtered audio signal
figure; % New figure for filtered audio
plot((1:length(audioFiltered))/fs, audioFiltered);
title('Filtered Audio Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Perform FFT on the filtered audio signal
audioFilteredFFT = fft(audioFiltered);
audioFilteredFFTShifted = fftshift(audioFilteredFFT); % Shift zero frequency component to center

% Calculate magnitude and focus only on the human voice range
magnitude = abs(audioFilteredFFTShifted);
freqRangeIdx = frequencies >= lowCutoff & frequencies <= highCutoff;
magnitudeInRange = magnitude(freqRangeIdx);
frequenciesInRange = frequencies(freqRangeIdx);

% Identify noise frequencies in the human voice range
threshold = mean(magnitudeInRange) * 0.5; % Set threshold to 50% of the average magnitude
noiseFrequencies = frequenciesInRange(magnitudeInRange < threshold);

% Create a notch filter to remove identified noise frequencies
for noiseFreq = noiseFrequencies
    % Design a notch filter around each identified noise frequency
    bw = 10; % Bandwidth of the notch filter (Hz)
    notchFreq = [noiseFreq - bw/2, noiseFreq + bw/2] / nyquist;
    [bNotch, aNotch] = butter(2, notchFreq, 'stop'); % 2nd order notch filter
    audioFiltered = filter(bNotch, aNotch, audioFiltered); % Apply notch filter
end

% Normalize the filtered audio to prevent clipping
audioFiltered = audioFiltered / max(abs(audioFiltered));

% Perform Fourier Transform on the final filtered audio signal
audioFinalFFT = fft(audioFiltered);
audioFinalFFTShifted = fftshift(audioFinalFFT); % Shift zero frequency component to center

% Plot frequency domain of final filtered audio signal
figure; % New figure for frequency domain of filtered audio
plot(frequencies - fs/2, abs(audioFinalFFTShifted));
title('Frequency Domain of Final Filtered Audio Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

% Save the final filtered audio to a new WAV file
outputFilePath = fullfile(pathname, 'filtered_audio.wav');
audiowrite(outputFilePath, audioFiltered, fs); % Save filtered audio
disp(['Filtered audio saved to: ', outputFilePath]);
