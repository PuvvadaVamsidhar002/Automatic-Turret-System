clc; clear; close all;

% Simulation Parameters
numSteps = 200;
dt = 1;
turretPosition = [0; 0];
bulletSpeed = 900;

% Noise Parameters
processNoise = 0.02;
measurementNoise = 0.8;
Q = eye(4) * processNoise;
R = eye(2) * measurementNoise;

% Generate Realistic Jet Path with Maneuvering
t = linspace(0, 10, numSteps);
truePositions = [5*t; 5*sin(0.5*t) + 2*sin(0.3*t) + randn(1, numSteps)];

% EKF Initialization
x = [truePositions(:,1); 1; 0];  % [x; y; vx; vy]
P = eye(4);                      % Initial covariance

% Models
f = @(x)[x(1) + dt * x(3);
         x(2) + dt * x(4);
         x(3);
         x(4)];

h = @(x)[x(1); x(2)];

% Jacobians (constant for linear motion model)
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
    % True Jet Position
    truePosition = truePositions(:, i);

    % Simulate Noisy Measurement
    z = truePosition + randn(2,1) * measurementNoise;
    measuredPositions(i,:) = z';

    % --- EKF Predict Step ---
    x_pred = f(x);
    P_pred = F * P * F' + Q;

    % --- EKF Update Step ---
    y = z - h(x_pred);           % Innovation
    S = H * P_pred * H' + R;     % Innovation covariance
    K = P_pred * H' / S;         % Kalman gain
    x = x_pred + K * y;          % Update state
    P = (eye(4) - K * H) * P_pred;

    predictedPositions(i,:) = x(1:2)';

    % Predictive Aiming (look-ahead based on past delta)
    if i > 5
        delta = predictedPositions(i,:) - predictedPositions(i-5,:);
        futurePosition = predictedPositions(i,:) + delta;
    else
        futurePosition = predictedPositions(i,:);
    end

    % Bullet Targeting
    direction = futurePosition - turretPosition';
    if norm(direction) > 0
        bulletTrajectory = turretPosition' + (direction / norm(direction)) * bulletSpeed * dt;
    else
        bulletTrajectory = turretPosition';
    end
    bulletPositions(i,:) = bulletTrajectory;

    % Hit Detection
    if norm(truePosition' - predictedPositions(i,:)) < 0.75
        hitPoints = [hitPoints; truePosition'];
    end

    % Plotting
    clf;
    hold on;
    plot(truePositions(1,1:i), truePositions(2,1:i), 'g-', 'LineWidth', 2);             % True path
    plot(predictedPositions(1:i,1), predictedPositions(1:i,2), 'm-', 'LineWidth', 2);   % EKF path
    plot(bulletPositions(1:i,1), bulletPositions(1:i,2), 'b--', 'LineWidth', 2);        % Bullet path
    if ~isempty(hitPoints)
        plot(hitPoints(:,1), hitPoints(:,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Hits
    end

    legend('True Jet Path', 'Predicted Path (EKF)', 'Bullet Path', 'Bullet Hits');
    xlabel('X Position'); ylabel('Y Position'); title(sprintf('Step: %d (Only EKF)', i));
    xlim([0, max(truePositions(1,:)) + 5]); ylim([-10, 10]);
    grid on;
    hold off;
    pause(0.1);
end
