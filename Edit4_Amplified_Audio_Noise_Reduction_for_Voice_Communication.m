% Advanced Audio Noise Reduction for Voice Communication
% Using Fourier Transform with Voice Enhancement

% Clear workspace and figures
clear; clc; close all;

% Run the main function
main_noise_reduction();

function [audioFiltered, fs] = advanced_noise_reduction(audioIn, fs)
    % Voice-focused Noise Reduction with Multiple Stages
    
    % Ensure column vector
    audioIn = audioIn(:);
    
    % Stage 1: FFT-based Noise Reduction with Voice Enhancement
    windowLength = round(0.05 * fs); % 50ms windows
    overlap = round(windowLength * 0.75);
    
    % Pad signal to handle edges
    paddingSize = floor(windowLength/2);
    paddedSignal = [zeros(paddingSize, 1); audioIn; zeros(paddingSize, 1)];
    
    % Initialize output signal
    audioFiltered = zeros(size(paddedSignal));
    window = hanning(windowLength);
    
    % Process signal in overlapping frames
    for i = 1:overlap:(length(paddedSignal)-windowLength+1)
        % Extract frame
        frame = paddedSignal(i:i+windowLength-1) .* window;
        
        % Apply FFT
        fftFrame = fft(frame);
        
        % Compute magnitude and phase
        magnitude = abs(fftFrame);
        phase = angle(fftFrame);
        
        % Enhanced voice frequency boost (typically 100-3000 Hz)
        freqRes = fs/length(fftFrame);
        voiceFreqIndices = round((100/freqRes):round(3000/freqRes));
        
        % Boost voice frequencies with a bell curve
        centerIdx = round(length(voiceFreqIndices)/2);
        boostFactor = 1.5 + 0.5 * exp(-(([1:length(voiceFreqIndices)]-centerIdx).^2)/(2*(centerIdx/4)^2));
        
        magnitude(voiceFreqIndices) = magnitude(voiceFreqIndices) .* boostFactor';
        
        % Simple noise estimation from high frequencies
        noiseLevel = mean(magnitude(floor(end*0.8):end)) * 0.75; % Reduced noise reduction
        
        % Spectral subtraction with less aggressive noise removal
        magnitudeProcessed = max(magnitude - noiseLevel, 0.3 * magnitude); % Increased minimum
        
        % Reconstruct frame
        processedFrame = real(ifft(magnitudeProcessed .* exp(1j * phase)));
        
        % Overlap-add
        audioFiltered(i:i+windowLength-1) = audioFiltered(i:i+windowLength-1) + processedFrame;
    end
    
    % Remove padding
    audioFiltered = audioFiltered(paddingSize+1:end-paddingSize);
    
    % Stage 2: Enhanced bandpass filter for voice frequencies
    nyquist = fs/2;
    freqLow = 80/nyquist;
    freqHigh = 3800/nyquist;
    filterOrder = 4;
    
    % Manual Butterworth filter implementation
    [b, a] = butterworth_bandpass(filterOrder, [freqLow freqHigh]);
    audioFiltered = filter(b, a, audioFiltered);
    
    % Stage 3: Dynamic range compression and voice enhancement
    frameLength = round(0.02 * fs); % 20ms frames
    numFrames = floor(length(audioFiltered)/frameLength);
    
    % Enhanced compression parameters
    threshold = 0.3;      % Lower threshold for more aggressive compression
    ratio = 0.6;         % Compression ratio (1/ratio for upward compression)
    makeupGain = 1.8;    % Makeup gain to boost overall level
    
    for i = 1:numFrames
        frameStart = (i-1)*frameLength + 1;
        frameEnd = min(frameStart + frameLength - 1, length(audioFiltered));
        frame = audioFiltered(frameStart:frameEnd);
        
        % RMS level for current frame
        rmsLevel = sqrt(mean(frame.^2));
        
        % Upward compression for quiet parts
        if rmsLevel < threshold
            gain = (threshold/rmsLevel)^ratio;
            gain = min(gain, 4.0); % Limit maximum gain
        else
            gain = 1.0;
        end
        
        % Apply gain with makeup gain
        frame = frame * gain * makeupGain;
        
        % Soft clipping to prevent harsh distortion
        frame = tanh(frame * 0.7) / 0.7;
        
        audioFiltered(frameStart:frameEnd) = frame;
    end
    
    % Stage 4: Final processing and normalization
    % Enhance presence frequencies (2-4 kHz) for voice clarity
    [b, a] = presenceBoostFilter(fs);
    audioFiltered = filter(b, a, audioFiltered);
    
    % Smooth any sudden amplitude changes
    smoothingWindow = ones(round(fs*0.01), 1) / round(fs*0.01); % 10ms window
    envelope = sqrt(conv(audioFiltered.^2, smoothingWindow, 'same'));
    smoothFactor = 0.95;
    audioFiltered = audioFiltered .* (smoothFactor + (1-smoothFactor) * (envelope / max(envelope)));
    
    % Final normalization with increased headroom
    maxAmp = max(abs(audioFiltered));
    if maxAmp > 0
        audioFiltered = audioFiltered / maxAmp * 0.95; % Increased from 0.9 to 0.95
    end
end

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

    % Display original audio signal
    figure('Name', 'Audio Analysis', 'Position', [100 100 1200 800]); 
    
    subplot(2,2,1);
    plot((1:length(audioIn))/fs, audioIn);
    title('Original Audio Signal');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
    
    % Original frequency spectrum
    subplot(2,2,2);
    [S, F] = custom_spectrogram(audioIn, fs);
    imagesc(F, 1:size(S,2), abs(S)');
    axis xy;
    title('Original Frequency Spectrum');
    xlabel('Frequency (Hz)');
    ylabel('Time Frame');
    colorbar;

    % Apply noise reduction
    [audioFiltered, ~] = advanced_noise_reduction(audioIn, fs);

    % Display filtered signal
    subplot(2,2,3);
    plot((1:length(audioFiltered))/fs, audioFiltered);
    title('Filtered Audio Signal');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;

    % Filtered frequency spectrum
    subplot(2,2,4);
    [S, F] = custom_spectrogram(audioFiltered, fs);
    imagesc(F, 1:size(S,2), abs(S)');
    axis xy;
    title('Filtered Frequency Spectrum');
    xlabel('Frequency (Hz)');
    ylabel('Time Frame');
    colorbar;

    % Save the filtered audio
    [~, name, ~] = fileparts(filename);
    outputFilePath = fullfile(pathname, [name '_filtered.wav']);
    audiowrite(outputFilePath, audioFiltered, fs);
    disp(['Filtered audio saved to: ', outputFilePath]);
end

function [b, a] = butterworth_bandpass(order, cutoff)
    % Manual implementation of Butterworth bandpass filter
    f1 = cutoff(1);
    f2 = cutoff(2);
    
    % Create lowpass filter
    [b1, a1] = butterworth_lowpass(order, f2);
    
    % Create highpass filter
    [b2, a2] = butterworth_highpass(order, f1);
    
    % Combine filters
    b = conv(b1, b2);
    a = conv(a1, a2);
end

function [b, a] = butterworth_lowpass(order, cutoff)
    % Simple Butterworth lowpass filter implementation
    omega = tan(pi * cutoff);
    omega2 = omega * omega;
    
    a = [1, omega/sqrt(2), omega2];
    b = omega2 * [1, 1, 1];
    
    % Normalize
    a = a / a(1);
    b = b / a(1);
end

function [b, a] = butterworth_highpass(order, cutoff)
    % Simple Butterworth highpass filter implementation
    omega = tan(pi * cutoff);
    omega2 = omega * omega;
    
    a = [1, omega/sqrt(2), omega2];
    b = [1, -2, 1];
    
    % Normalize
    a = a / a(1);
    b = b / a(1);
end

function [b, a] = presenceBoostFilter(fs)
    % Design a filter to boost presence frequencies (2-4 kHz)
    f0 = 3000; % Center frequency
    Q = 1.0;   % Quality factor
    gain = 2;  % Gain in dB
    
    w0 = 2*pi*f0/fs;
    alpha = sin(w0)/(2*Q);
    A = 10^(gain/40);
    
    b = [1 + alpha*A, -2*cos(w0), 1 - alpha*A];
    a = [1 + alpha/A, -2*cos(w0), 1 - alpha/A];
end

function [S, F] = custom_spectrogram(signal, fs)
    % Custom spectrogram implementation using FFT
    windowLength = 1024;
    overlap = 512;
    nfft = 2048;
    
    % Create Hanning window
    window = 0.5 * (1 - cos(2*pi*(0:windowLength-1)'/(windowLength-1)));
    
    % Calculate number of frames
    numFrames = floor((length(signal) - overlap)/(windowLength - overlap));
    
    % Initialize spectrogram matrix
    S = zeros(nfft/2+1, numFrames);
    
    % Process each frame
    for i = 1:numFrames
        % Extract frame
        startIdx = (i-1)*(windowLength-overlap) + 1;
        frame = signal(startIdx:startIdx+windowLength-1) .* window;
        
        % Compute FFT
        X = fft(frame, nfft);
        
        % Store positive frequencies
        S(:,i) = X(1:nfft/2+1);
    end
    
    % Generate frequency vector
    F = (0:nfft/2)' * fs/nfft;
end