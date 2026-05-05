%% Adaptive Controller — Pose Regulation Simulation
% Slotine-Li adaptive controller per Siciliano Section 8.5.
% Regulates robot to constant desired joint configuration from initial
% conditions q0. Plant uses TRUE parameters p_true (30% inertia mismatch).
% Controller adapts pi_hat starting from nominal p(7:22).
clc, clear all, close all;

%% Paths
addpath("Communication_Code");
addpath("generated_dynamics");

%% Timing
t_sample = 0.002;       % simulation timestep [s]
tfin     = 4;          % longer than trajectory — regulation needs settling time
t        = 0:t_sample:tfin;
N        = length(t);

%% System parameters — nominal (used by controller and regressor)
R  = load('Identification/identification_result.mat');
p  = [R.p(1:6); ...
      R.x_opt_vec(1);  R.x_opt_vec(2);  R.x_opt_vec(3);  R.x_opt_vec(4); ...
      R.x_opt_vec(5);  R.x_opt_vec(6);  R.x_opt_vec(7); ...
      R.x_opt_vec(8);  R.x_opt_vec(9);  R.x_opt_vec(10); ...
      R.x_opt_vec(11); R.x_opt_vec(12); R.x_opt_vec(13); ...
      R.x_opt_vec(14); R.x_opt_vec(15); R.x_opt_vec(16); ...
      R.id_info.g];
pf = [R.x_opt_vec(17); R.x_opt_vec(18); R.x_opt_vec(19); R.x_opt_vec(20)];

%% True plant parameters — 30% inertia mismatch to test adaptation
p_true        = p;
p_true(11:22) = p_true(11:22) * 1.3;

%% Initial and desired configurations
q0    = [0.1;  0.1;  0.1;  0.0];       % initial joint positions [rad]
q_des = [0.50; -0.35; 0.30; 0.15];     % desired setpoint [rad]

% Constant desired trajectory (zero velocity and acceleration)
qd_traj   = repmat(q_des,  1, N);
dqd_traj  = zeros(4, N);
ddqd_traj = zeros(4, N);

%% Controller gains (Siciliano 8.5)
% Lambda: sets position error weight inside s = de + Lambda*e
% Kd:     gain on filtered error s in control law tau = Y*pi_hat + Kd*s
% Gamma:  adaptation rate
lambda_joints = [5.0; 9.0; 7.0; 7.0];
Lambda = diag(lambda_joints);

% Kd scaled by nominal inertia at desired config for physical consistency
M0   = M_fun(q_des, p);
m    = diag(M0);
zeta = [1.0; 1.0; 1.0; 1.0];       % critically damped for regulation
Kd   = diag(2 * zeta .* lambda_joints .* m);

Gamma = diag(0.01 * ones(16,1));    % adaptation rate

%% Print effective closed-loop bandwidths
fprintf('--- Effective bandwidths ---\n');
for i = 1:4
    fprintf('Joint %d: lambda=%.2f | Kd=%.4f\n', i, lambda_joints(i), Kd(i,i));
end

%% State storage
q      = zeros(4, N+1);
q_dot  = zeros(4, N+1);
tau_k  = zeros(4, N);
pi_k   = zeros(16, N+1);

%% Initial conditions
q(:,1)    = q0;
pi_k(:,1) = p(7:22);   % start from nominal identified parameters

%% Main simulation loop
for k = 1:N
    q_k    = q(:,k);
    dq_k   = q_dot(:,k);
    qd_k   = qd_traj(:,k);     % constant setpoint
    dqd_k  = dqd_traj(:,k);    % zero
    ddqd_k = ddqd_traj(:,k);   % zero

    %% Adaptive control law
    [tau, pi_next] = adaptive_controller(p, ...
        q_k, qd_k, dq_k, dqd_k, ddqd_k, ...
        pi_k(:,k), Kd, Lambda, Gamma, t_sample);

    tau_k(:,k)  = tau;
    pi_k(:,k+1) = pi_next;

    %% RK4 integration — plant uses TRUE parameters p_true
    % pi_hat frozen at pi_k(:,k) during substeps (correct per Siciliano)
    ctrl = @(x) adaptive_controller( p,...
        x(1:4), qd_k, x(5:8), dqd_k, ddqd_k, ...
        pi_k(:,k), Kd, Lambda, Gamma, t_sample);

    f = @(x) [ ...
        x(5:8); ...
        M_fun(x(1:4), p_true) \ ( ...
            ctrl(x) ...
            - C_fun(x(1:4), x(5:8), p_true) * x(5:8) ...
            - G_fun(x(1:4), p_true) ...
            - ViscousFriction_fun(x(5:8), pf) ...
        ) ...
    ];

    h  = t_sample;
    xk = [q_k; dq_k];

    k1 = f(xk);
    k2 = f(xk + 0.5*h*k1);
    k3 = f(xk + 0.5*h*k2);
    k4 = f(xk + h*k3);

    x_next = xk + (h/6)*(k1 + 2*k2 + 2*k3 + k4);

    if any(~isfinite(x_next))
        fprintf('DIVERGED at k=%d\n', k);
        break
    end

    q(:,k+1)     = x_next(1:4);
    q_dot(:,k+1) = x_next(5:8);
end

%% Trim to N steps
q_k     = q(:,     1:N);
q_dot_k = q_dot(:, 1:N);

%% Error norm ||e(t)||_2
e_norm = vecnorm(qd_traj - q_k, 2, 1);

%% Plotting

% (a) Joint positions
figure(1)
for i = 1:4
    subplot(2,2,i)
    plot(t, rad2deg(qd_traj(i,:)), 'r-', 'LineWidth', 1.5); hold on
    plot(t, rad2deg(q_k(i,:)),     'b.-', 'MarkerSize', 3)
    xlabel('Time [s]'); ylabel(['q', num2str(i), ' [deg]'])
    title(['Joint ', num2str(i), ' Position'])
    legend('Desired','Actual','Location','best'); grid on
end
sgtitle('Adaptive Controller — Joint Positions (Regulation)')

% (b) Joint velocities
figure(2)
for i = 1:4
    subplot(2,2,i)
    plot(t, rad2deg(q_dot_k(i,:)), 'c.-', 'MarkerSize', 3)
    xlabel('Time [s]'); ylabel(['dq', num2str(i), ' [deg/s]'])
    title(['Joint ', num2str(i), ' Velocity']); grid on
end
sgtitle('Adaptive Controller — Joint Velocities (Regulation)')

% (c) Tracking errors
figure(3)
for i = 1:4
    subplot(2,2,i)
    plot(t, rad2deg(qd_traj(i,:) - q_k(i,:)), 'g.-', 'MarkerSize', 3)
    yline(0, '--k')
    xlabel('Time [s]'); ylabel(['e', num2str(i), ' [deg]'])
    title(['Joint ', num2str(i), ' Error']); grid on
end
sgtitle('Adaptive Controller — Tracking Errors (Regulation)')

% (d) Error norm
figure(4)
plot(t, e_norm, 'r.-', 'LineWidth', 1.5)
xlabel('Time [s]'); ylabel('||e||_2 [rad]')
title('Euclidean Norm of Joint Error'); grid on
sgtitle('Adaptive Controller — Error Norm (Regulation)')

% (e) Control torques
figure(5)
for i = 1:4
    subplot(2,2,i)
    plot(t, tau_k(i,:), 'm.-', 'MarkerSize', 3)
    xlabel('Time [s]'); ylabel(['\tau', num2str(i), ' [Nm]'])
    title(['Joint ', num2str(i), ' Torque']); grid on
end
sgtitle('Adaptive Controller — Control Torques (Regulation)')

% (f) Parameter estimates evolution
% Red dashed line = true nominal value from identification
figure(6)
for i = 1:16
    subplot(4,4,i)
    plot(t, pi_k(i,1:N), 'b.-', 'LineWidth', 1); hold on
    yline(p(6+i), '--r', 'LineWidth', 1)
    xlabel('Time [s]')
    title(['\pi_{', num2str(i), '}']); grid on
end
sgtitle('Adaptive Controller — Parameter Estimates \hat{\pi} (Regulation)')
% Add shared legend via annotation
annotation('textbox',[0.85 0.95 0.1 0.04],'String','— Estimated  -- Nominal', ...
    'EdgeColor','none','FontSize',8);