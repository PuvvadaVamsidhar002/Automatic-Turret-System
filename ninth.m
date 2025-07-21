clc; clear; close all;

% Simulation Parameters
numSteps = 200;    
dt = 1;            
turretPosition = [0; 0; 0]; % Fixed turret position
bulletSpeed = 900;          % Adjusted for realistic bullet velocity

% Noise parameters
processNoise = 0.01;  
measurementNoise = 0.3; 

% Generate a Realistic Fighter Jet Path in 3D
t = linspace(0, 10, numSteps);
truePositions = [
    5*t; 
    5*sin(0.5*t) + 2*sin(1.5*t) + randn(1, numSteps); % More erratic Y movement [CHANGED]
    2*cos(0.4*t) + 1.5*cos(2*t) + randn(1, numSteps)  % More erratic Z movement [CHANGED]
];  % Smooth, controlled maneuvers

% Kalman Filter Initialization for 3D
kalmanFilter = vision.KalmanFilter(...
    'StateTransitionModel', [1 0 0 dt 0 0; 0 1 0 0 dt 0; 0 0 1 0 0 dt; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1], ...
    'MeasurementModel', [1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0], ...
    'ProcessNoise', eye(6) * processNoise, ...
    'MeasurementNoise', eye(3) * measurementNoise, ...
    'State', [truePositions(:,1); 1; 0; 0]);

% Data Storage
predictedPositions = zeros(numSteps, 3);
trackedPositions = zeros(numSteps, 3);
measuredPositions = zeros(numSteps, 3);
bulletPositions = NaN(numSteps, 3);
hitPoints = []; 
shotsFired = 0;
shotsHit = 0;

% Savitzky-Golay Filtering for Advanced Smoothing
windowSize = 11; 
polynomialOrder = 2; 

figure; hold on;
for i = 1:numSteps
    % True Jet Position
    truePosition = truePositions(:, i);
    
    % Simulate Noisy Sensor Measurement
    measuredPosition = truePosition + randn(3,1) * measurementNoise;
    measuredPositions(i, :) = measuredPosition';

    % Kalman Filter Prediction & Correction
    predictedState = predict(kalmanFilter);
    predictedPositions(i, :) = predictedState(1:3)';
    correctedState = correct(kalmanFilter, measuredPosition);
    trackedPositions(i, :) = correctedState(1:3)';

    % Adaptive Process Noise
    if i > 1 && norm(measuredPositions(i,:) - measuredPositions(i-1,:)) > 3
        kalmanFilter.ProcessNoise = eye(6) * 0.1;
    else
        kalmanFilter.ProcessNoise = eye(6) * processNoise;
    end

    % Apply Savitzky-Golay Filter for better smoothing
    if i > windowSize
        smoothedPredictions = sgolayfilt(predictedPositions(1:i,:), polynomialOrder, windowSize);
    else
        smoothedPredictions = predictedPositions(1:i,:);
    end

    % Bullet Targeting System with Predictive Aiming
    if i > 5
        futurePosition = smoothedPredictions(end,:) + 1.5*(smoothedPredictions(end,:) - smoothedPredictions(end-5,:));
    else
        futurePosition = smoothedPredictions(end,:);
    end
    direction = futurePosition - turretPosition';
    if norm(direction) > 0
        bulletTrajectory = turretPosition' + (direction / norm(direction)) * bulletSpeed * dt;
    else
        bulletTrajectory = turretPosition';
    end
    bulletPositions(i, :) = bulletTrajectory;
    shotsFired = shotsFired + 1;

    % Bullet Hit Detection
    if norm(truePosition' - smoothedPredictions(end, :)) < 0.75
        hitPoints = [hitPoints; truePosition']; 
        shotsHit = shotsHit + 1;
    end

    % Plot Results in 3D
    clf;
    hold on;
    plot3(truePositions(1,1:i), truePositions(2,1:i), truePositions(3,1:i), 'g-', 'LineWidth', 2); % True Jet Path
    plot3(smoothedPredictions(:,1), smoothedPredictions(:,2), smoothedPredictions(:,3), 'm-', 'LineWidth', 2); % Smoothed Path
    plot3(bulletPositions(1:i,1), bulletPositions(1:i,2), bulletPositions(1:i,3), 'b--', 'LineWidth', 2); % Bullet Path
    if ~isempty(hitPoints)
        plot3(hitPoints(:,1), hitPoints(:,2), hitPoints(:,3), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Bullet Hits
    end

    % Display turret position
    plot3(turretPosition(1), turretPosition(2), turretPosition(3), 'kx', 'MarkerSize', 10, 'LineWidth', 2);

    % Display stats
    text(0, -10, 10, sprintf('Shots Fired: %d', shotsFired), 'Color', 'k', 'FontSize', 12);
    text(0, -10, 8, sprintf('Shots Hit: %d', shotsHit), 'Color', 'r', 'FontSize', 12);
    text(0, -10, 6, sprintf('Accuracy: %.2f%%', (shotsHit / shotsFired) * 100), 'Color', 'b', 'FontSize', 12);

    legend('True Jet Path', 'Predicted Path (Smoothed)', 'Bullet Path', 'Bullet Hits', 'Turret Position');
    xlabel('X Position'); ylabel('Y Position'); zlabel('Z Position'); title(sprintf('Step: %d', i));
    xlim([0, max(truePositions(1,:)) + 5]); ylim([-15, 15]); zlim([-15, 15]); % Adjusted limits for crazy movement [CHANGED]
    grid on;
    view(3);
    hold off;
    pause(0.1);
end