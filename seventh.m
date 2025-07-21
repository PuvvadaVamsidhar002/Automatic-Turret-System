clc; clear; close all;

% Simulation Parameters
numSteps = 200;    
dt = 1;            
turretPosition = [0; 0]; 
bulletSpeed = 1000;   % Adjusted to realistic anti-aircraft speed

% Adaptive Noise
processNoise = 0.02;  
measurementNoise = 0.8; 

% Generate a Realistic Jet Path (Evasive Maneuvers)
t = linspace(0, 10, numSteps);
truePositions = [5*t + 3*sin(0.5*t); 5*sin(0.5*t) + 2*randn(1, numSteps)];

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
totalBulletsFired = numSteps;
totalHits = 0;  % Counter for successful hits

% Savitzky-Golay Filtering for Advanced Smoothing
windowSize = 11; 
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

    % Adaptive Process Noise (Increases when large jumps occur)
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

    % Bullet Targeting System - Predict Future Jet Position
    if i > 2
        velocity = (trackedPositions(i,:) - trackedPositions(i-1,:)) / dt;
        acceleration = (trackedPositions(i,:) - 2*trackedPositions(i-1,:) + trackedPositions(i-2,:)) / dt^2;
        futurePosition = trackedPositions(i,:) + velocity * 0.2 + 0.5 * acceleration * 0.2^2; % Predict 0.2s ahead
    else
        velocity = [0, 0];  % Fix: Initialize velocity for i <= 2
        futurePosition = trackedPositions(i,:);
    end

    % Bullet Movement
    direction = futurePosition - turretPosition';
    if norm(direction) > 0
        bulletTrajectory = turretPosition' + (direction / norm(direction)) * bulletSpeed * dt;
    else
        bulletTrajectory = turretPosition';
    end
    bulletPositions(i, :) = bulletTrajectory;

    % **Improved Hit Detection (Near-Miss Zone)**
    hitRadius = max(0.75, norm(velocity) / 50); % Fix: Avoid using uninitialized velocity
    if norm(truePosition' - bulletTrajectory) < hitRadius
        hitPoints = [hitPoints; truePosition']; 
        totalHits = totalHits + 1;
        fprintf("Hit at (%.2f, %.2f) at step %d\n", truePosition(1), truePosition(2), i);
    end

    % Plot Results
    clf;
    hold on;
    plot(truePositions(1,1:i), truePositions(2,1:i), 'g-', 'LineWidth', 2); % True Jet Path
    plot(smoothedPredictions(:,1), smoothedPredictions(:,2), 'm-', 'LineWidth', 2); % Smoothed Path
    plot(bulletPositions(1:i,1), bulletPositions(1:i,2), 'b--', 'LineWidth', 2); % Bullet Path
    if ~isempty(hitPoints)
        plot(hitPoints(:,1), hitPoints(:,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Bullet Hits
    end

    legend('True Jet Path', 'Predicted Path (Smoothed)', 'Bullet Path', 'Bullet Hits');
    xlabel('X Position'); ylabel('Y Position'); title(sprintf('Step: %d', i));
    xlim([0, max(truePositions(1,:)) + 5]); ylim([-10, 10]);
    grid on;
    hold off;
    pause(0.05);
end

% **Display Final Hit Count & Accuracy**
accuracy = (totalHits / totalBulletsFired) * 100;
fprintf("\nTotal Bullets Fired: %d\n", totalBulletsFired);
fprintf("Total Hits: %d\n", totalHits);
fprintf("Final Accuracy: %.2f%%\n", accuracy);
