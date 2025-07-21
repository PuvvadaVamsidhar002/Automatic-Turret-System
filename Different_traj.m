clc; clear; close all;

% Simulation Parameters
numSteps = 200;    
dt = 1;          
hits = 0;
shotsfired = 0;
turretPosition = [0; 0]; 
bulletSpeed = 1000;   

% Initial noise values
initialProcessNoise = 0.0075;  
measurementNoise = 0.1; 

% Time vector
t = linspace(0, 10, numSteps);

% Choose a pattern for the jet's motion
pattern = 'accelerating'; % Options: 'sine', 'straight', 'circular', 'random', 'zigzag', 'accelerating'

% Generate true positions based on the selected pattern
switch pattern
    case 'sine'
        % Sine wave motion
        truePositions = [5*t; 5*sin(0.5*t)];
    case 'straight'
        % Straight line with constant velocity
        truePositions = [5*t; 2*ones(1, numSteps)];
    case 'circular'
        % Circular motion
        radius = 10;
        angularVelocity = 0.5;
        truePositions = [radius * cos(angularVelocity * t); radius * sin(angularVelocity * t)];
    case 'random'
        % Random motion
        truePositions = cumsum(randn(2, numSteps), 2);
    case 'zigzag'
        % Zigzag motion
        frequency=0.5;
        amplitude=5;
        truePositions=[5*t; amplitude*sawtooth(frequency*t)];
    case 'accelerating'
        % Accelerating motion
        truePositions = [0.1*t.^2; 5*sin(0.5*t)];
    otherwise
        error('Invalid pattern selected.');
end

% Kalman Filter Initialization
kalmanFilter = vision.KalmanFilter(...
    'StateTransitionModel', [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1], ...
    'MeasurementModel', [1 0 0 0; 0 1 0 0], ...
    'ProcessNoise', eye(4) * initialProcessNoise, ...
    'MeasurementNoise', eye(2) * measurementNoise, ...
    'State', [truePositions(:,1); 1; 0]);

% Data Storage
predictedPositions = zeros(numSteps, 2);
trackedPositions = zeros(numSteps, 2);
measuredPositions = zeros(numSteps, 2);
bulletPositions = NaN(numSteps, 2);
hitPoints = []; 

% Adaptive Filtering Parameters
residualThreshold = 1.0; % Threshold to adjust process noise
maxProcessNoise = 0.1;   % Maximum process noise
minProcessNoise = 0.01;  % Minimum process noise

% Savitzky-Golay Filtering for Advanced Smoothing
windowSize = 11; % Choose a moderate smoothing window
polynomialOrder = 2; 

figure; hold on;
for i = 1:numSteps
    % True Jet Position
    truePosition = truePositions(:, i);
    
    % Simulate Noisy Sensor Measurement
    measuredPosition = truePosition + randn(2,1) * measurementNoise;
    measuredPositions(i, :) = measuredPosition';

    % Kalman Filter Prediction & Correction
    predictedState = predict(kalmanFilter);
    predictedPositions(i, :) = predictedState(1:2)';
    correctedState = correct(kalmanFilter, measuredPosition);
    trackedPositions(i, :) = correctedState(1:2)';

    % Calculate Residual (Difference between predicted and measured position)
    residual = norm(measuredPosition - predictedState(1:2));

    % Adaptive Process Noise Adjustment
    if residual > residualThreshold
        % Increase process noise if residual is large
        kalmanFilter.ProcessNoise = eye(4) * maxProcessNoise;
    else
        % Decrease process noise if residual is small
        kalmanFilter.ProcessNoise = eye(4) * minProcessNoise;
    end

    % Apply Savitzky-Golay Filter for better smoothing
    if i > windowSize
        smoothedPredictions = sgolayfilt(predictedPositions(1:i,:), polynomialOrder, windowSize);
    else
        smoothedPredictions = predictedPositions(1:i,:);
    end

    % Bullet Targeting System
    direction = smoothedPredictions(end,:) - turretPosition';
    if norm(direction) > 0
        bulletTrajectory = turretPosition' + (direction / norm(direction)) * bulletSpeed * dt;
    else
        bulletTrajectory = turretPosition';
    end
    bulletPositions(i, :) = bulletTrajectory;

    % Bullet Hit Detection (improved precision)
    if norm(truePosition' - smoothedPredictions(end, :)) < 0.75
        hitPoints = [hitPoints; truePosition']; 
        hits = hits + 1;
    end
    shotsfired = shotsfired + 1;

    % Plot Results
    clf;
    hold on;
    plot(truePositions(1,1:i), truePositions(2,1:i), 'g-', 'LineWidth', 2); % True Jet Path
    plot(smoothedPredictions(:,1), smoothedPredictions(:,2), 'm-', 'LineWidth', 2); % Smoothed Path
    plot(bulletPositions(1:i,1), bulletPositions(1:i,2), 'b--', 'LineWidth', 2); % Bullet Path
    if ~isempty(hitPoints)
        plot(hitPoints(:,1), hitPoints(:,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Bullet Hits
    end
   % Replace the text calls with annotation
    annotation('textbox', [0.15, 0.8, 0.1, 0.1], 'String', sprintf('Shots Fired: %d', shotsfired), 'Color', 'k', 'FontSize', 12, 'EdgeColor', 'none');
    annotation('textbox', [0.15, 0.75, 0.1, 0.1], 'String', sprintf('Shots Hit: %d', hits), 'Color', 'r', 'FontSize', 12, 'EdgeColor', 'none');
    annotation('textbox', [0.15, 0.7, 0.1, 0.1], 'String', sprintf('Accuracy: %.2f%%', (hits / shotsfired) * 100), 'Color', 'b', 'FontSize', 12, 'EdgeColor', 'none');

    legend('True Jet Path', 'Predicted Path (Smoothed)', 'Bullet Path', 'Bullet Hits');
    xlabel('X Position'); ylabel('Y Position'); title(sprintf('Step: %d, Pattern: %s', i, pattern));
    xlim([min(truePositions(1,:))-5, max(truePositions(1,:))+5]);
    ylim([min(truePositions(2,:))-5, max(truePositions(2,:))+5]);
    grid on;
    hold off;
    pause(0.1);
end