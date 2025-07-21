clc; clear; close all;

% Create 3D environment
figure;
axis([-200 200 -200 200 0 300]); % Define space limits (X, Y, Z)
grid on; hold on;
xlabel('X (meters)'); ylabel('Y (meters)'); zlabel('Z (meters)');
view(3); % 3D perspective
%% 
jet_pos = [0, 0, 100]; % Initial jet position
jet = plot3(jet_pos(1), jet_pos(2), jet_pos(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
%% 
time_steps = 100; % Number of frames in simulation

for t = 1:time_steps
    % Define motion: Jet moving forward and oscillating
    jet_pos = [t - 50, 30 * sin(t / 10), 100 + 10 * sin(t / 5)];
    
    % Update jet position
    set(jet, 'XData', jet_pos(1), 'YData', jet_pos(2), 'ZData', jet_pos(3));
    
    drawnow; % Refresh figure
    pause(0.1); % Delay for animation
end
%% 
turret_pos = [0, 0, 0]; % Turret is fixed at origin
plot3(turret_pos(1), turret_pos(2), turret_pos(3), 'bo', 'MarkerSize', 12, 'MarkerFaceColor', 'b');
%% 
for t = 1:time_steps
    % Update jet position
    jet_pos = [t - 50, 30 * sin(t / 10), 100 + 10 * sin(t / 5)];
    set(jet, 'XData', jet_pos(1), 'YData', jet_pos(2), 'ZData', jet_pos(3));

    % Calculate direction vector from turret to jet
    dx = jet_pos(1) - turret_pos(1);
    dy = jet_pos(2) - turret_pos(2);
    dz = jet_pos(3) - turret_pos(3);

    % Plot turret aiming line
    if exist('aim_line', 'var')
        delete(aim_line);
    end
    aim_line = plot3([turret_pos(1), jet_pos(1)], ...
                      [turret_pos(2), jet_pos(2)], ...
                      [turret_pos(3), jet_pos(3)], 'k--', 'LineWidth', 2);

    drawnow;
    pause(0.1);
end
%% 
for t = 1:time_steps
    jet_pos = [t - 50, 30 * sin(t / 10), 100 + 10 * sin(t / 5)];
    set(jet, 'XData', jet_pos(1), 'YData', jet_pos(2), 'ZData', jet_pos(3));

    dx = jet_pos(1) - turret_pos(1);
    dy = jet_pos(2) - turret_pos(2);
    dz = jet_pos(3) - turret_pos(3);

    if exist('aim_line', 'var')
        delete(aim_line);
    end
    aim_line = plot3([turret_pos(1), jet_pos(1)], ...
                      [turret_pos(2), jet_pos(2)], ...
                      [turret_pos(3), jet_pos(3)], 'k--', 'LineWidth', 2);

    % Bullet motion from turret to jet
    bullet = plot3(turret_pos(1), turret_pos(2), turret_pos(3), 'go', 'MarkerSize', 6, 'MarkerFaceColor', 'g');
    for step = 1:20
        bullet_pos = turret_pos + (step / 20) * [dx, dy, dz]; % Linear interpolation
        set(bullet, 'XData', bullet_pos(1), 'YData', bullet_pos(2), 'ZData', bullet_pos(3));
        drawnow;
        pause(0.05);
    end
    delete(bullet); % Remove bullet after it reaches target

    pause(0.1);
end
%% 
[F, V] = stlread('jet_model.stl'); % Load STL file
jet_model = patch('Faces', F, 'Vertices', V, 'FaceColor', 'red');
%% 
camtarget(jet_pos); % Set camera to follow jet
    





