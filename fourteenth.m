clc; clear; close all;

% Simulation Parameters
numSteps = 200;
dt = 1;
hits = 0;
shotsfired = 0;
turretPosition = [0; 0];
bulletSpeed = 1000;

% Noise parameters
processNoise = 0.07;
measurementNoise = 0.8;
Q = eye(4) * processNoise;
R = eye(2) * measurementNoise;

% Jet trajectory
t = linspace(0, 10, numSteps);
truePositions = [5*t; 5*sin(0.5*t)];

% EKF Initialization
x = [truePositions(:,1); 1; 0];  % [x; y; vx; vy]
P = eye(4);                     % Initial covariance

% Models
f = @(x)[x(1) + dt * x(3);
         x(2) + dt * x(4);
         x(3);
         x(4)];
h = @(x)[x(1); x(2)];

% Jacobians
F = [1 0 dt 0;
     0 1 0 dt;
     0 0 1  0;
     0 0 0  1];
H = [1 0 0 0;
     0 1 0 0];

% Data Storage
predictedPositions = zeros(numSteps, 2);
measuredPositions = zeros(numSteps, 2);
bulletPositions = NaN(numSteps, 2);
hitPoints = [];

figure; hold on;
for i = 1:numSteps
    % Ground truth
    truePosition = truePositions(:, i);

    % Simulated noisy measurement
    z = truePosition + randn(2,1) * measurementNoise;
    measuredPositions(i,:) = z';

    % EKF Predict
    x_pred = f(x);
    P_pred = F * P * F' + Q;

    % EKF Update
    y = z - h(x_pred);
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;
    x = x_pred + K * y;
    P = (eye(4) - K * H) * P_pred;

    % Store prediction
    predictedPositions(i,:) = x(1:2)';

    % Bullet Firing
    direction = predictedPositions(i,:) - turretPosition';
    if norm(direction) > 0
        bulletTrajectory = turretPosition' + (direction / norm(direction)) * bulletSpeed * dt;
    else
        bulletTrajectory = turretPosition';
    end
    bulletPositions(i,:) = bulletTrajectory;

    % Hit Detection
    if norm(truePosition' - predictedPositions(i,:)) < 0.75
        hitPoints = [hitPoints; truePosition'];
        hits = hits + 1;
    end
    shotsfired = shotsfired + 1;

    % Plot
    clf;
    hold on;
    plot(truePositions(1,1:i), truePositions(2,1:i), 'g-', 'LineWidth', 2);
    plot(predictedPositions(1:i,1), predictedPositions(1:i,2), 'm-', 'LineWidth', 2);
    plot(bulletPositions(1:i,1), bulletPositions(1:i,2), 'b--', 'LineWidth', 2);
    if ~isempty(hitPoints)
        plot(hitPoints(:,1), hitPoints(:,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    end
    text(40, 2, 14, sprintf('Shots Fired: %d', shotsfired), 'Color', 'k', 'FontSize', 12);
    text(40, 1, 10, sprintf('Shots Hit: %d', hits), 'Color', 'r', 'FontSize', 12);
    text(40, 0, 6, sprintf('Accuracy: %.2f%%', (hits / shotsfired) * 100), 'Color', 'b', 'FontSize', 12);

    legend('True Jet Path', 'Predicted Path (EKF)', 'Bullet Path', 'Bullet Hits');
    xlabel('X Position'); ylabel('Y Position');
    title(sprintf('Step: %d (Only EKF)', i));
    xlim([0, max(truePositions(1,:)) + 5]); ylim([-10, 10]);
    grid on;
    hold off;
    pause(0.1);
end
