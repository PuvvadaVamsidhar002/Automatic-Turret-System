clc; clear; close all;
% Simulation Parameters
numSteps = 200;    
dt = 1;          
hits=0;
shotsfired=0;
turretPosition = [0; 0]; 
bulletSpeed = 1000;   

% Reduce erratic movements by fine-tuning noise
processNoise = 0.0075;  
measurementNoise = 0.1; 
t = linspace(0, 10, numSteps);
truePositions = [5*t; 5*sin(0.5*t)];

%rng(42); % Set seed for reproducibility
%truePositions = zeros(2, numSteps);
%truePositions(:,1) = [0; 0]; % Start position

%for i = 2:numSteps
    % Add random movement with a slight preference for forward motion
 %   deltaX = 5 + randn * 0.5; % Mostly moves forward, with some randomness
  %  deltaY = randn * 1.5; % Random up/down movement

   % truePositions(:, i) = truePositions(:, i-1) + [deltaX; deltaY];
%end

% Kalman Filter Initialization
kalmanFilter = vision.KalmanFilter(...
    'StateTransitionModel', [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1], ...
    'MeasurementModel', [1 0 0 0; 0 1 0 0], ...
    'ProcessNoise', eye(4) * processNoise, ...
    'MeasurementNoise', eye(2) * measurementNoise, ...
    'State', [truePositions(:,1); 1; 0]);

% Data Storage
predictedPositions = zeros(numSteps, 2);
trackedPositions = zeros(numSteps, 2);
measuredPositions = zeros(numSteps, 2);
bulletPositions = NaN(numSteps, 2);
hitPoints = []; 

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

    % Adaptive Process Noise (increases when large jumps occur)
    if i > 1 && norm(measuredPositions(i,:) - measuredPositions(i-1,:)) > 3
        kalmanFilter.ProcessNoise = eye(4) * 0.1;
    else
        kalmanFilter.ProcessNoise = eye(4) * processNoise;
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
        hits=hits+1;
    end
    shotsfired=shotsfired+1;
    % Plot Results
    clf;
    hold on;
    plot(truePositions(1,1:i), truePositions(2,1:i), 'g-', 'LineWidth', 2); % True Jet Path
    plot(smoothedPredictions(:,1), smoothedPredictions(:,2), 'm-', 'LineWidth', 2); % Smoothed Path
    plot(bulletPositions(1:i,1), bulletPositions(1:i,2), 'b--', 'LineWidth', 2); % Bullet Path
    if ~isempty(hitPoints)
        plot(hitPoints(:,1), hitPoints(:,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Bullet Hits
    end
    text(40, 2, 14, sprintf('Shots Fired: %d', shotsfired), 'Color', 'k', 'FontSize', 12);
    text(40, 1, 10, sprintf('Shots Hit: %d', hits), 'Color', 'r', 'FontSize', 12);
    text(40, 0, 6, sprintf('Accuracy: %.2f%%', (hits / shotsfired) * 100), 'Color', 'b', 'FontSize', 12);

    legend('True Jet Path', 'Predicted Path (Smoothed)', 'Bullet Path', 'Bullet Hits');
    xlabel('X Position'); ylabel('Y Position'); title(sprintf('Step: %d', i));
    xlim([0, max(truePositions(1,:)) + 5]); ylim([-10, 10]);
    grid on;
    hold off;
    pause(0.1);
end
