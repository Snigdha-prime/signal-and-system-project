% Advanced Audio Noise Reduction with Visualization

clear; clc; close all;

% main function
main_noise_reduction();

function main_noise_reduction()
    % Load the audio file
    [filename, pathname] = uigetfile({'*.wav; *.mp3', 'Audio Files (*.wav, *.mp3)'}, 'Select an audio file');
    if isequal(filename, 0) || isequal(pathname, 0)
        disp('User canceled the file selection.');
        return;
    end
    audioFilePath = fullfile(pathname, filename);
    [audioIn, fs] = audioread(audioFilePath);

    % Convert to mono if stereo
    if size(audioIn, 2) > 1
        audioIn = mean(audioIn, 2);
    end

    % Apply noise reduction
    audioFiltered = advanced_noise_reduction(audioIn, fs);

    % Save the filtered audio
    [~, name, ~] = fileparts(filename);
    outputFilePath = fullfile(pathname, [name '_filtered.wav']);
    audiowrite(outputFilePath, audioFiltered, fs);
    disp(['Filtered audio saved to: ', outputFilePath]);

    % Generate figures
    generate_figures(audioIn, audioFiltered, fs);
end

function audioFiltered = advanced_noise_reduction(audioIn, fs)
    % Ensure column vector
    audioIn = audioIn(:);
    
    % Stage 1: FFT-based Noise Reduction
    fftFiltered = fft_noise_reduction(audioIn, fs);
    
    % Stage 2: Wavelet-Based Denoising
    waveletFiltered = wavelet_denoising(fftFiltered, fs);

    % Stage 3: High-Decibel Attenuation
    audioFiltered = high_decibel_attenuation(waveletFiltered, fs);
    
    % Final normalization
    audioFiltered = audioFiltered / max(abs(audioFiltered));
end

function audioFiltered = fft_noise_reduction(audioIn, fs)
    % FFT-based noise reduction
    
    windowLength = round(0.05 * fs); % 50ms windows
    overlap = round(windowLength * 0.75);
    paddedSignal = [zeros(floor(windowLength / 2), 1); audioIn; zeros(floor(windowLength / 2), 1)];
    audioFiltered = zeros(size(paddedSignal));
    window = hanning(windowLength);
    
    for i = 1:overlap:(length(paddedSignal) - windowLength + 1)
        frame = paddedSignal(i:i + windowLength - 1) .* window;
        fftFrame = fft(frame);
        magnitude = abs(fftFrame);
        phase = angle(fftFrame);
        
        freqRes = fs / length(fftFrame);
        voiceFreqIndices = round((100 / freqRes):(3000 / freqRes));
        magnitude(voiceFreqIndices) = magnitude(voiceFreqIndices) * 1.5;
        noiseLevel = mean(magnitude(floor(end * 0.75):end)) * 0.5;
        magnitudeProcessed = max(magnitude - noiseLevel, 0.2 * magnitude);
        processedFrame = real(ifft(magnitudeProcessed .* exp(1j * phase)));
        audioFiltered(i:i + windowLength - 1) = audioFiltered(i:i + windowLength - 1) + processedFrame;
    end
    
    audioFiltered = audioFiltered(floor(windowLength / 2) + 1:end - floor(windowLength / 2));
end

function denoisedAudio = wavelet_denoising(audioIn, fs)
    % Wavelet-based denoising
    waveletName = 'db6';
    decompositionLevel = 5;
    [C, L] = wavedec(audioIn, decompositionLevel, waveletName);
    sigma = median(abs(C(L(1):end))) / 0.6745;
    threshold = sigma * sqrt(2 * log(length(audioIn)));
    denoisedC = wthresh(C, 's', threshold);
    denoisedAudio = waverec(denoisedC, L, waveletName);
end

function audioOut = high_decibel_attenuation(audioIn, fs)
    % High-decibel attenuation
    dbThreshold = -20;
    linearThreshold = 10^(dbThreshold / 20);
    windowLength = round(0.05 * fs);
    energy = sqrt(movmean(audioIn.^2, windowLength));
    attenuationFactor = 0.5;
    audioOut = audioIn;
    audioOut(energy > linearThreshold) = audioOut(energy > linearThreshold) * attenuationFactor;
end

function generate_figures(audioIn, audioFiltered, fs)
    % Generate visualization figures

    % 1. Waveform Comparison
    figure;
    subplot(2, 1, 1);
    plot((1:length(audioIn)) / fs, audioIn);
    title('Original Audio Waveform');
    xlabel('Time (s)');
    ylabel('Amplitude');
    subplot(2, 1, 2);
    plot((1:length(audioFiltered)) / fs, audioFiltered);
    title('Filtered Audio Waveform');
    xlabel('Time (s)');
    ylabel('Amplitude');

    % 2. Spectrogram Comparison
    figure;
    subplot(2, 1, 1);
    spectrogram(audioIn, 256, 128, 256, fs, 'yaxis');
    title('Spectrogram of Original Audio');
    subplot(2, 1, 2);
    spectrogram(audioFiltered, 256, 128, 256, fs, 'yaxis');
    title('Spectrogram of Filtered Audio');

    % 3. FFT Magnitude Spectrum
    figure;
    freqAxis = linspace(0, fs / 2, length(audioIn) / 2);
    fftOriginal = abs(fft(audioIn));
    fftFiltered = abs(fft(audioFiltered));
    plot(freqAxis, fftOriginal(1:length(freqAxis)), 'b', 'DisplayName', 'Original');
    hold on;
    plot(freqAxis, fftFiltered(1:length(freqAxis)), 'r', 'DisplayName', 'Filtered');
    hold off;
    legend;
    title('FFT Magnitude Spectrum');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');

    % 4. Wavelet Coefficients
    waveletName = 'db6';
    decompositionLevel = 5;
    [COriginal, LOriginal] = wavedec(audioIn, decompositionLevel, waveletName);
    [CFiltered, LFiltered] = wavedec(audioFiltered, decompositionLevel, waveletName);
    figure;
    subplot(2, 1, 1);
    plot(COriginal);
    title('Wavelet Coefficients - Original Audio');
    subplot(2, 1, 2);
    plot(CFiltered);
    title('Wavelet Coefficients - Filtered Audio');

    % 5. Energy/Decibel Attenuation
    windowLength = round(0.05 * fs);
    energyOriginal = sqrt(movmean(audioIn.^2, windowLength));
    energyFiltered = sqrt(movmean(audioFiltered.^2, windowLength));
    timeAxis = (1:length(energyOriginal)) / fs;
    figure;
    plot(timeAxis, 20 * log10(energyOriginal), 'b', 'DisplayName', 'Original');
    hold on;
    plot(timeAxis, 20 * log10(energyFiltered), 'r', 'DisplayName', 'Filtered');
    hold off;
    legend;
    title('Energy (dB) Over Time');
    xlabel('Time (s)');
    ylabel('Energy (dB)');
end
