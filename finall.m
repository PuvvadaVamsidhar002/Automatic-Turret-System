clc; clear; close all;

% Simulation Parameters
numSteps = 200;    
dt = 1;            
turretPosition = [0; 0; 0]; % Fixed turret position
bulletSpeed = 900; % Bullet velocity

% Noise parameters
processNoise = 0.01;  
measurementNoise = 0.2; 

% Generate Realistic Fighter Jet Paths in 3D for 3 Jets with random starting positions
t = linspace(0, 10, numSteps);
jetStartPositions = 5 * randn(3, 3); % Random starting positions for each jet (3 jets, 3 coordinates each)

truePositions = cat(3, [5*t + jetStartPositions(1,1); 5*sin(0.5*t) + jetStartPositions(2,1); 2*cos(0.2*t) + jetStartPositions(3,1)], ...  % Jet 1
                      [5*t + jetStartPositions(1,2); 5*cos(0.2*t) + jetStartPositions(2,2); 2*sin(0.4*t) + jetStartPositions(3,2)], ...  % Jet 2
                      [5*t + jetStartPositions(1,3); 5*sin(0.7*t) + jetStartPositions(2,3); 2*cos(0.6*t) + jetStartPositions(3,3)]);  % Jet 3

% Kalman Filter Initialization for 3D
numJets = 3;
kalmanFilters = cell(1, numJets);
for j = 1:numJets
    kalmanFilters{j} = vision.KalmanFilter(...
        'StateTransitionModel', [1 0 0 dt 0 0; 0 1 0 0 dt 0; 0 0 1 0 0 dt; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1], ...
        'MeasurementModel', [1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0], ...
        'ProcessNoise', eye(6) * processNoise, ...
        'MeasurementNoise', eye(3) * measurementNoise, ...
        'State', [truePositions(:,1,j); 1; 0; 0]);
end

% Data Storage
predictedPositions = zeros(numSteps, 3, numJets);
trackedPositions = zeros(numSteps, 3, numJets);
measuredPositions = zeros(numSteps, 3, numJets);
bulletPositions = NaN(numSteps, 3);
hitPoints = [];
shotsFired = 0;
shotsHit = 0;
hitsOnJet = zeros(1, numJets);
jetDestroyed = false(1, numJets);
currentTarget = 1;

% Jet Shape (Triangle)
jetVertices = [-1 0 0; 1 0 0; 0 1.5 0]; 
jetFaces = [1 2 3];

% Create figure with improved presentation
figure('Color', 'white', 'Position', [100, 100, 1000, 800]);
hold on;
grid on;
axis equal;
xlabel('X Position (m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Y Position (m)', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('Z Position (m)', 'FontSize', 12, 'FontWeight', 'bold');
title('3D Air Defense Simulation', 'FontSize', 16, 'FontWeight', 'bold');
view(3);

% Custom colors
jetColors = [1 0.5 0.5; 0.5 1 0.5; 0.5 0.5 1]; % Light red, green, blue for jets
bulletColor = [0 0.4470 0.7410]; % Blue for bullets
hitColor = [1 0 0]; % Red for hits
turretColor = [0 0 0]; % Black for turret

for i = 1:numSteps
    if all(jetDestroyed)
        break; % Stop simulation if all jets are destroyed
    end

    % Check if the current target is destroyed, switch to next
    if jetDestroyed(currentTarget)
        availableTargets = find(~jetDestroyed);
        if ~isempty(availableTargets)
            currentTarget = availableTargets(1);
        else
            break;
        end
    end

    for j = 1:numJets
        if jetDestroyed(j)
            continue;
        end

        % True Jet Position
        truePosition = truePositions(:, i, j);

        % Simulate Noisy Sensor Measurement
        measuredPosition = truePosition + randn(3,1) * measurementNoise;
        measuredPositions(i, :, j) = measuredPosition';

        % Kalman Filter Prediction & Correction
        predictedState = predict(kalmanFilters{j});
        predictedPositions(i, :, j) = predictedState(1:3)';
        correctedState = correct(kalmanFilters{j}, measuredPosition);
        trackedPositions(i, :, j) = correctedState(1:3)';
    end

    % Targeting only the current jet
    futurePosition = predictedPositions(i, :, currentTarget);
    direction = futurePosition - turretPosition';
    if norm(direction) > 0
        bulletTrajectory = turretPosition' + (direction / norm(direction)) * bulletSpeed * dt;
    else
        bulletTrajectory = turretPosition';
    end
    bulletPositions(i, :) = bulletTrajectory;
    shotsFired = shotsFired + 1;

    % Bullet Hit Detection
    if norm(truePositions(:, i, currentTarget)' - predictedPositions(i, :, currentTarget)) < 0.75
        hitPoints = [hitPoints; truePositions(:, i, currentTarget)'];
        shotsHit = shotsHit + 1;
        hitsOnJet(currentTarget) = hitsOnJet(currentTarget) + 1;

        if hitsOnJet(currentTarget) > 20
            jetDestroyed(currentTarget) = true;
        end
    end

    % Plot Results in 3D with improved presentation
    clf;
    hold on;
    grid on;
    
    % Set the view and axis limits
    xlim([0, max(truePositions(1,:)) + 5]);
    ylim([min(truePositions(2,:))-5, max(truePositions(2,:))+5]);
    zlim([min(truePositions(3,:))-5, max(truePositions(3,:))+5]);
    view(3);

    % Plot initial positions with specified colors and larger markers
    h_j1_start = plot3(truePositions(1,1,1), truePositions(2,1,1), truePositions(3,1,1), ...
         'o', 'Color', [1 0.5 0.5], 'MarkerSize', 10, 'MarkerFaceColor', [1 0.5 0.5]);
    h_j2_start = plot3(truePositions(1,1,2), truePositions(2,1,2), truePositions(3,1,2), ...
         'o', 'Color', [0.5 1 0.5], 'MarkerSize', 10, 'MarkerFaceColor', [0.5 1 0.5]);
    h_j3_start = plot3(truePositions(1,1,3), truePositions(2,1,3), truePositions(3,1,3), ...
         'o', 'Color', [0.5 0.5 1], 'MarkerSize', 10, 'MarkerFaceColor', [0.5 0.5 1]);

    % Jet Visualization with different colors
    h_jets = gobjects(1, numJets);
    h_destroyed = gobjects(1, numJets);
    for j = 1:numJets
        if ~jetDestroyed(j)
            h_jets(j) = patch('Vertices', jetVertices + truePositions(:, i, j)', 'Faces', jetFaces, ...
                 'FaceColor', jetColors(j,:), 'EdgeColor', 'k', 'LineWidth', 1);
        else
            h_destroyed(j) = plot3(truePositions(1, i, j), truePositions(2, i, j), truePositions(3, i, j), ...
                 'rx', 'MarkerSize', 15, 'LineWidth', 3);
        end
    end

    % Bullet path with improved style
    h_bullet = plot3(bulletPositions(1:i,1), bulletPositions(1:i,2), bulletPositions(1:i,3), ...
         'Color', bulletColor, 'LineStyle', '--', 'LineWidth', 2, 'Marker', 'o', 'MarkerSize', 4);

    % Hit markers (red dots)
    if ~isempty(hitPoints)
        h_hits = scatter3(hitPoints(:,1), hitPoints(:,2), hitPoints(:,3), ...
                'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
    else
        h_hits = [];
    end

    % Turret visualization (black X)
    h_turret = plot3(turretPosition(1), turretPosition(2), turretPosition(3), ...
         'Marker', 'x', 'Color', 'k', 'MarkerSize', 15, 'LineWidth', 3);

    % Display stats in a more professional way
    statsText = {sprintf('Simulation Step: %d', i), ...
                sprintf('Shots Fired: %d', shotsFired), ...
                sprintf('Shots Hit: %d', shotsHit), ...
                sprintf('Accuracy: %.2f%%', (shotsHit / shotsFired) * 100)};
            
    text(0, min(ylim), max(zlim), statsText, ...
        'FontSize', 12, 'BackgroundColor', 'white', 'EdgeColor', 'k', 'Margin', 2);

    % Jet status indicators
    jetStatus = {};
    for j = 1:numJets
        if jetDestroyed(j)
            jetStatus{end+1} = sprintf('\\color{red}Jet %d DESTROYED', j);
        else
            jetStatus{end+1} = sprintf('\\color[rgb]{%f,%f,%f}Jet %d Active (Hits: %d/20)', ...
                                      jetColors(j,1), jetColors(j,2), jetColors(j,3), j, hitsOnJet(j));
        end
    end
    
    text(max(xlim)*0.6, min(ylim), max(zlim), jetStatus, ...
        'FontSize', 12, 'BackgroundColor', 'white', 'EdgeColor', 'k', 'Margin', 2, ...
        'Interpreter', 'tex');

    % Current target indicator
    if ~jetDestroyed(currentTarget)
        text(truePositions(1,i,currentTarget), truePositions(2,i,currentTarget), truePositions(3,i,currentTarget), ...
            'TARGET', 'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end

    % Create legend with all elements
    legendItems = {
        'Jet 1 Initial Point', ...
        'Jet 2 Initial Point', ...
        'Jet 3 Initial Point', ...
        'Jet 1', ...
        'Jet 2', ...
        'Jet 3', ...
        'Bullets', ...
        'Bullet Hit Points', ...
        'Turret Position', ...
        'Jet 1 Destroyed', ...
        'Jet 2 Destroyed', ...
        'Jet 3 Destroyed'
        };
    
    % Create legend
    legend(legendItems, 'Location', 'northeast', 'FontSize', 10, 'NumColumns', 2);

    drawnow;
    pause(0.1);
end

% Final summary
fprintf('\n=== Simulation Results ===\n');
fprintf('Total Shots Fired: %d\n', shotsFired);
fprintf('Total Shots Hit: %d\n', shotsHit);
fprintf('Overall Accuracy: %.2f%%\n', (shotsHit/shotsFired)*100);
fprintf('Jets Destroyed: %d out of %d\n', sum(jetDestroyed), numJets);
for j = 1:numJets
    fprintf('Jet %d: %s\n', j, ternary(jetDestroyed(j), 'DESTROYED', 'survived'));
end