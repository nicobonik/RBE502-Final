%% OpenManipulator-X forward dynamics simulation
clc, clear all, close all;

%% Paths
addpath("Communication_Code");
addpath("generated_dynamics");

%% Timing
t_sample = 0.002;
tfin = 10;
t = 0:t_sample:tfin;
N = length(t);


%% State variables
q = zeros(4, N+1);
q_dot = zeros(4, N+1);
tau_k = zeros(4, N);
pi_k = zeros(16, N+1);
dt = zeros(1, N);

%% Desired states for each joint
qd = [0.5; -0.35; 0.3; 0.15];
dqd = [0; 0; 0; 0];
ddqd = [0; 0; 0; 0];

%% System parameters
R =load('Identification/identification_result.mat');
disp(R.p(1:6))
p = [R.p(1:6); ...
     R.x_opt_vec(1); R.x_opt_vec(2); R.x_opt_vec(3); R.x_opt_vec(4); ...
     R.x_opt_vec(5); R.x_opt_vec(6); R.x_opt_vec(7); ...
     R.x_opt_vec(8); R.x_opt_vec(9); R.x_opt_vec(10); ...
     R.x_opt_vec(11); R.x_opt_vec(12); R.x_opt_vec(13); ...
     R.x_opt_vec(14); R.x_opt_vec(15); R.x_opt_vec(16); ...
     R.id_info.g];
pf = [R.x_opt_vec(17); R.x_opt_vec(18); R.x_opt_vec(19); R.x_opt_vec(20)];

p_true = p;
p_true(11:22) = p_true(11:22)  * 1.3;

q(:, 1) = [0.1 0.1 0.1 0]';
pi_k(:, 1) = p(7:22);

q_top = deg2rad([0;  0; -45;  45]);   % EE at top of pole
q_bot = deg2rad([0; 40;  45; -85]);   % EE at bottom of pole

[qd_traj, dqd_traj, ddqd_traj, t, ee_desired] = generate_traj_NW(q_top, q_bot, p, t_sample, tfin, false);
%% Main simulation loop

%% Parameters

lambda_joints = [5.0; 9.0; 7.0; 7.0];      % per-joint Lambda diagonal
Lambda = diag(lambda_joints);
Gamma  = diag(0.01 * ones(16,1));  % adaptation - tune per parameter group

M0 = M_fun(q(:, 1), p);
m  = diag(M0);
zeta = [1.0; 0.5; 1.0; 1.0];
Kd   = diag(2 * zeta .* lambda_joints .* m);


%% Initial conditions
q(:,1)      = q_top;
pi_k(:,1)   = p(7:22);

%% Main loop
for k = 1:N

   q_k  = q(:,k);
    dq_k = q_dot(:,k);

    % Current desired state from trajectory
    qd_k   = qd_traj(:,k);
    dqd_k  = dqd_traj(:,k);
    ddqd_k = ddqd_traj(:,k);

    % Adaptive controller with time-varying desired trajectory
    [tau, pi_next] = adaptive_controller(p, q_k, qd_k, dq_k, ...
                                          dqd_k, ddqd_k, pi_k(:,k), ...
                                          Kd, Lambda, Gamma, t_sample);
    tau_k(:,k)   = tau;
    pi_k(:,k+1)  = pi_next;

    %% RK4 - recompute tau at each substep
    ctrl = @(x) adaptive_controller(p, x(1:4), qd_k, x(5:8), dqd_k, ddqd_k, ...
                                     pi_k(:, k), Kd, Lambda, Gamma, t_sample);

    f = @(x) [ ...
        x(5:8); ...
        M_fun(x(1:4), p_true) \ ( ...
            ctrl(x) ...
            - C_fun(x(1:4), x(5:8), p_true)*x(5:8) ...
            - G_fun(x(1:4), p_true) ...
            - ViscousFriction_fun(x(5:8), pf) ...
        ) ...
    ];

    %% Fixed timestep
    h  = t_sample;
    xk = [q_k; dq_k];

    k1 = f(xk);
    k2 = f(xk + 0.5*h*k1);
    k3 = f(xk + 0.5*h*k2);
    k4 = f(xk + h*k3);

    x_next = xk + (h/6)*(k1 + 2*k2 + 2*k3 + k4);

    q(:,k+1)     = x_next(1:4);
    q_dot(:,k+1) = x_next(5:8);
end

E = repmat(qd, 1, N) - q(:,1:N);

ee_actual = zeros(3, N);
for k = 1:N
    ee_actual(:,k) = FK_fun(q(:,k), p);
end

figure;
for i = 1:4
    subplot(4,1,i);
    plot(t, q(i,1:N), 'b', t, qd(i)*ones(1,N), 'r--', 'LineWidth', 1.5);
    ylabel(['q_' num2str(i) ' (rad)']);
    legend('Actual','Desired'); grid on;
end
xlabel('Time (s)'); sgtitle('Joint Positions');

figure;
for i = 1:4
    subplot(4,1,i);
    plot(t, q_dot(i,1:N), 'b', 'LineWidth', 1.5);
    ylabel(['\dot{q}_' num2str(i) ' (rad/s)']); grid on;
end
xlabel('Time (s)'); sgtitle('Joint Velocities');

figure;
for i = 1:4
    subplot(4,1,i);
    plot(t, E(i,:), 'r', 'LineWidth', 1.5);
    ylabel(['e_' num2str(i) ' (rad)']); grid on;
end
xlabel('Time (s)'); sgtitle('Tracking Error');

figure;
for i = 1:4
    subplot(4,1,i);
    plot(t, tau_k(i,:), 'k', 'LineWidth', 1.5);
    ylabel(['\tau_' num2str(i) ' (Nm)']); grid on;
end
xlabel('Time (s)'); sgtitle('Control Torques');

%% Joint tracking
figure;
for i = 1:4
    subplot(4,1,i);
    plot(t, q(i,1:N),     'b',  'LineWidth', 1.5); hold on;
    plot(t, qd_traj(i,:), 'r--','LineWidth', 1.5);
    ylabel(['q_' num2str(i) ' (rad)']);
    legend('Actual','Desired'); grid on;
end
sgtitle('Joint Position Tracking'); xlabel('Time (s)');

figure;
for i = 1:4
    subplot(4,1,i);
    plot(t, q_dot(i, 1:N), 'b', 'LineWidth', 1.5); hold on;
    plot(t, dqd_traj(i, :), 'r--', 'LineWidth', 1.5);
    ylabel(['dq_' num2str(i) ' (rad/s)']);
    legend('Actual','Desired'); grid on;
end
sgtitle('Joint Velocity Tracking'); xlabel('Time (s)');

figure;
for i = 1:16
    subplot(4,4,i);
    plot(t, pi_k(i, 1:N), 'b', 'LineWidth', 1.5);
    ylabel(['pi_{' num2str(i) '}']);
    grid on;
end
sgtitle('Model Parameters');

%% Cartesian tracking - most important for straight line
figure;
labels = {'X (m)', 'Y (m)', 'Z (m)'};
for i = 1:3
    subplot(3,1,i);
    plot(t, ee_actual(i,:),  'b',  'LineWidth', 1.5); hold on;
    plot(t, ee_desired(i,:), 'r--','LineWidth', 1.5);
    ylabel(labels{i}); legend('Actual','Desired'); grid on;
end
sgtitle('End Effector Cartesian Tracking'); xlabel('Time (s)');

% %% 3D trajectory visualization
% figure;
% plot3(ee_desired(1,:), ee_desired(2,:), ee_desired(3,:), ...
%       'r--', 'LineWidth', 2); hold on;
% plot3(ee_actual(1,:),  ee_actual(2,:),  ee_actual(3,:),  ...
%       'b',   'LineWidth', 1.5);
% xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
% legend('Desired straight line', 'Actual path');
% title('3D End Effector Path'); grid on; axis equal;

figure;
plot(ee_desired(1,:), ee_desired(3,:), ...
     'r--', 'LineWidth', 2); hold on;
plot(ee_actual(1,:),  ee_actual(3,:),  ...
     'b',   'LineWidth', 1.5);
scatter(ee_desired(1,1),   ee_desired(3,1),   100, 'g', 'filled');  % start
scatter(ee_desired(1,end), ee_desired(3,end), 100, 'k', 'filled');  % end
xlabel('X (m)'); ylabel('Z (m)');
legend('Desired straight line', 'Actual path', 'Start', 'End');
title('End Effector Path (X-Z Plane)'); 
grid on; axis equal;


%% Cartesian error - how straight is the line?
cart_error = vecnorm(ee_actual - ee_desired, 2, 1);
figure;
plot(t, cart_error*1000, 'r', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Cartesian Error (mm)');
title('End Effector Path Error'); grid on;
fprintf('Max cartesian error:  %.2f mm\n', max(cart_error)*1000);
fprintf('Mean cartesian error: %.2f mm\n', mean(cart_error)*1000);
