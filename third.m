clc; clear; close all;

% Simulation Parameters
numSteps = 100;    
dt = 1;            
turretPosition = [0; 0]; 
bulletSpeed = 1000;   
processNoise = 0.07; 
measurementNoise = 0.8; 

% Hit Tracking
hits = 0;
shotsfired = 0;

% Generate Smooth Jet Path
t = linspace(0, 10, numSteps);
truePositions = [5*t; 5*sin(0.5*t)]; 

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

figure; hold on;
for i = 1:numSteps
    % True Jet Position
    truePosition = truePositions(:, i);
    
    % Simulate Noisy Sensor Measurement
    measuredPosition = truePosition + randn(2,1) * measurementNoise;
    measuredPositions(i, :) = measuredPosition';

    % Kalman Filter Prediction
    predictedState = predict(kalmanFilter);
    predictedPositions(i, :) = predictedState(1:2)';

    % Kalman Filter Correction
    correctedState = correct(kalmanFilter, measuredPosition);
    trackedPositions(i, :) = correctedState(1:2)';

    % Fire Bullet from Turret to Predicted Jet Position
    direction = predictedPositions(i, :) - turretPosition';
    if norm(direction) > 0
        bulletTrajectory = turretPosition' + (direction / norm(direction)) * bulletSpeed * dt;
    else
        bulletTrajectory = turretPosition';
    end
    bulletPositions(i, :) = bulletTrajectory;

    % Check if Bullet Hits the Jet
    if norm(truePosition' - predictedPositions(i, :)) < 1
        hitPoints = [hitPoints; truePosition']; 
        hits = hits + 1;
    end
    shotsfired = shotsfired + 1;

    % Plot Results
    clf;
    hold on;
    plot(truePositions(1,1:i), truePositions(2,1:i), 'g-', 'LineWidth', 2); % True Jet Path
    plot(predictedPositions(1:i,1), predictedPositions(1:i,2), 'm-', 'LineWidth', 2); % Predicted Path
    plot(bulletPositions(1:i,1), bulletPositions(1:i,2), 'b--', 'LineWidth', 2); % Bullet Path
    if ~isempty(hitPoints)
        plot(hitPoints(:,1), hitPoints(:,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Bullet Hits
    end

    % Add Hit Stats as Text
    text(40, 2, 14, sprintf('Shots Fired: %d', shotsfired), 'Color', 'k', 'FontSize', 12);
    text(40, 1, 10, sprintf('Shots Hit: %d', hits), 'Color', 'r', 'FontSize', 12);
    text(40, 0, 6, sprintf('Accuracy: %.2f%%', (hits / shotsfired) * 100), 'Color', 'b', 'FontSize', 12);

    % Add Legend
    legend('True Jet Path', 'Predicted Path', 'Bullet Path', 'Bullet Hits');

    xlabel('X Position'); ylabel('Y Position'); 
    title(sprintf('Step: %d', i));
    xlim([0, max(truePositions(1,:)) + 5]); 
    ylim([-10, 10]);
    grid on;
    hold off;
    pause(0.1);
end
