clc; clear; close all;
number=0
% Simulation Parameters
numSteps = 200;    
dt = 1;            
turretPosition = [0; 0]; 
bulletSpeed = 3;   

% Noise Parameters
processNoise = 0.02;  
measurementNoise = 0.8; 

% Generate Two Jet Paths (Two Sine Waves)
t = linspace(0, 10, numSteps);
truePositions1 = [5*t; 5*sin(0.5*t)];      % Plane 1
truePositions2 = [5*t+6; 5*sin(0.5*t)]; % Plane 2 (Phase shift)

% Compute the Midpoint of the Two Planes
midPositions = (truePositions1 + truePositions2) / 2;

% Kalman Filter Initialization (Tracking the Midpoint)
kalmanFilter = vision.KalmanFilter(...
    'StateTransitionModel', [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1], ...
    'MeasurementModel', [1 0 0 0; 0 1 0 0], ...
    'ProcessNoise', eye(4) * processNoise, ...
    'MeasurementNoise', eye(2) * measurementNoise, ...
    'State', [midPositions(:,1); 1; 0]);

% Data Storage
predictedPositions = zeros(numSteps, 2);
trackedPositions = zeros(numSteps, 2);
measuredPositions = zeros(numSteps, 2);
bulletPositions = NaN(numSteps, 2);
hitPoints = []; 

% Savitzky-Golay Filtering for Advanced Smoothing
windowSize = 11;
polynomialOrder = 2; 

figure; hold on;
for i = 1:numSteps
    % True Jet Positions
    truePosition1 = truePositions1(:, i);
    truePosition2 = truePositions2(:, i);
    
    % Compute Midpoint
    trueMidpoint = (truePosition1 + truePosition2) / 2;
    
    % Simulate Noisy Sensor Measurement
    measuredPosition = trueMidpoint + randn(2,1) * measurementNoise;
    measuredPositions(i, :) = measuredPosition';

    % Kalman Filter Prediction & Correction
    predictedState = predict(kalmanFilter);
    predictedPositions(i, :) = predictedState(1:2)';
    correctedState = correct(kalmanFilter, measuredPosition);
    trackedPositions(i, :) = correctedState(1:2)';

    % Adaptive Process Noise
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

    % Bullet Targeting System (Aiming at the Midpoint)
    direction = smoothedPredictions(end,:) - turretPosition';
    if norm(direction) > 0
        bulletTrajectory = turretPosition' + (direction / norm(direction)) * bulletSpeed * dt;
    else
        bulletTrajectory = turretPosition';
    end
    bulletPositions(i, :) = bulletTrajectory;

    % Bullet Hit Detection
    if norm(trueMidpoint' - smoothedPredictions(end, :)) < 0.75
        hitPoints = [hitPoints; trueMidpoint']; 
        number=number+1
    end

    % Plot Results
    clf;
    hold on;
    plot(truePositions1(1,1:i), truePositions1(2,1:i), 'r-', 'LineWidth', 2); % Plane 1 Path
    plot(truePositions2(1,1:i), truePositions2(2,1:i), 'b-', 'LineWidth', 2); % Plane 2 Path
    plot(smoothedPredictions(:,1), smoothedPredictions(:,2), 'm-', 'LineWidth', 2); % Midpoint Smoothed Path
    plot(bulletPositions(1:i,1), bulletPositions(1:i,2), 'g--', 'LineWidth', 2); % Bullet Path
    if ~isempty(hitPoints)
        plot(hitPoints(:,1), hitPoints(:,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Bullet Hits
    end

    legend('Plane 1 Path', 'Plane 2 Path', 'Midpoint Path (Target)', 'Bullet Path', 'Bullet Hits');
    xlabel('X Position'); ylabel('Y Position'); title(sprintf('Step: %d', i));
    xlim([0, max(truePositions1(1,:)) + 5]); ylim([-10, 10]);
    grid on;
    hold off;
    pause(0.1);
end
fprintf('Number of points matched: %d\nTotal numeber of points is: %d\nSo accuracy is:%f',number,numSteps,(number/numSteps)*100)