%% Computed Torque (Inverse Dynamics) Controller for OpenManipulator-X
clc, clear all, close all;

%% Paths
addpath("OpenManipulator-X/");
addpath("OpenManipulator-X/Communication_Code");
addpath("OpenManipulator-X/generated_dynamics");

%% Timing
t_sample = 0.01;        % sampling time [s]
tfin     = 4;          % final time [s]
t        = 0:t_sample:tfin;
N        = length(t);

%% Conversion factors
factor_degre_to_rad = pi/180;
factor_rad_to_degre = 180/pi;
factor_mA_to_A = 1/1000;
factor_A_to_mA = 1000/1;

%% Define robot
robot = Robot();

%% Define the time, this is time for a interpolation between where the robot is now and where you want to be
%% Do not decrease less than 1 since this can produce agressive movement
travelTime = 4.0; 

%% Define the type of low level controller, this is position control for each joint
robot.writeMode('p');

%% This defines the time for the interpolation.
%% Yous shoudl define this if you are using position control
robot.writeTime(travelTime); 

%% Activate the torque
robot.writeMotorState(true); 

disp("Initializing...")

%% Send zero to each joint this is Home of the robot
robot.writeJoints([0.1,0.1,0.1,0.0] * factor_rad_to_degre); 
pause(travelTime); 

%% Define the type of low level control of the robot this is current mode
robot.writeMode('c');

%% Joint Positions
q_real = zeros(4, length(t)+1);
q_dot_real = zeros(4, length(t)+1);
current_real = zeros(4, length(t)+1);

%% Read Initial Conditions
joint_readings = robot.getJointsReadings();
q_real(:, 1) = joint_readings(1, :)*factor_degre_to_rad;
q_dot_real(:, 1) = joint_readings(2, :)*factor_degre_to_rad;
current_real(:, 1) = joint_readings(3, :)*factor_mA_to_A;

%% Constants
q1_desired = 0.50  * ones(1, N);
q2_desired = -0.35 * ones(1, N);
q3_desired = 0.30  * ones(1, N);
q4_desired = 0.15  * ones(1, N);

q_desired = [q1_desired; q2_desired; q3_desired; q4_desired];

q_desired_dot = zeros(4, N);        % constant setpoint => zero velocity
q_desired_ddot = zeros(4, N);       % constant setpoint => zero acceleration

%% If you implement a full inverse dynamics controller you can define a desired velocity for each joint
% q_desired_dot = [0*q1_desired; 0*q2_desired; 0*q3_desired; 0*q4_desired];

%% Controller gains
% kp = wn^2,  kv = 2*wn
% Change Kp & Kv for different responses
wn = 8;
% Kp = (wn^2) * eye(4);
% Kv = (2*wn) * eye(4);

% Kp = 64 * eye(4);
% Kv = 16.2 * eye(4);

% wn_joints = [8, 9.5, 19, 8];
wn_joints = [8, 9.5, 19, 8.5];
Kp = diag(wn_joints.^2);
Kv = diag(2*wn_joints);

%% System parameters
R =load('OpenManipulator-X/Identification/identification_result.mat');
p = [R.p(1:6); ...
     R.x_opt_vec(1); R.x_opt_vec(2); R.x_opt_vec(3); R.x_opt_vec(4); ...
     R.x_opt_vec(5); R.x_opt_vec(6); R.x_opt_vec(7); ...
     R.x_opt_vec(8); R.x_opt_vec(9); R.x_opt_vec(10); ...
     R.x_opt_vec(11); R.x_opt_vec(12); R.x_opt_vec(13); ...
     R.x_opt_vec(14); R.x_opt_vec(15); R.x_opt_vec(16); ...
     R.id_info.g];
pf = [R.x_opt_vec(17); R.x_opt_vec(18); R.x_opt_vec(19); R.x_opt_vec(20)];

%% Initial conditions (from HW6)
q0     = [0.1; 0.1; 0.1; 0.0];
q_dot0 = [0;   0;   0;   0  ];

%% State storage
% q_real     = zeros(4, N+1);
% q_dot_real = zeros(4, N+1);
tau_k      = zeros(4, N);
dt         = zeros(1, N);

q_real(:,1)     = q0;
q_dot_real(:,1) = q_dot0;

%% Control Loop
for k = 1:length(t) 
    tic

    %% DEBUG: print current state
    fprintf('k=%3d | q=[%6.3f %6.3f %6.3f %6.3f] rad\n', ...
        k, q_real(1,k), q_real(2,k), q_real(3,k), q_real(4,k));

    %% Create Control Law Your Controller goes Here
    tau_k(:,k) = inverse_controller( ...
        q_real(:,k), q_dot_real(:,k), ...
        q_desired(:,k), q_desired_dot(:,k), q_desired_ddot(:,k), ...
        Kp, Kv, p);

    torques = [tau_k(1, k), tau_k(2, k), tau_k(3, k), tau_k(4, k)];
    %% This is the mapping to Amperes
    current = torque_to_current(torques);

    %% This is the mapping to mA
    current_mA = current*factor_A_to_mA;

    robot.writeCurrents(current_mA);

    while toc < t_sample
    end

    %% DEBUG
    e = q_desired(:,k) - q_real(:,k);
    fprintf('k=%3d | e=[%6.3f %6.3f %6.3f %6.3f] | tau=[%6.3f %6.3f %6.3f %6.3f] | I_mA=[%6.1f %6.1f %6.1f %6.1f]\n', ...
        k, e(1),e(2),e(3),e(4), ...
        tau_k(1,k),tau_k(2,k),tau_k(3,k),tau_k(4,k), ...
        current_mA(1),current_mA(2),current_mA(3),current_mA(4));

    %% Sample time
    dt(k) = toc;

    %% Update measurements
    joint_readings = robot.getJointsReadings();
    q_real(:, k+1) = joint_readings(1, :)*factor_degre_to_rad;
    q_dot_real(:, k+1) = joint_readings(2, :)*factor_degre_to_rad;
    current_real(:, k+1) = joint_readings(3, :)*factor_mA_to_A;

end

%% Save data histories at sample k
q_real_k = q_real(:, 1:length(t));
q_dot_real_k = q_dot_real(:, 1:length(t));
current_real_k = current_real(:, 1:length(t));

%% Final Values
tau = [0,0,0,0];
current = tau;
robot.writeCurrents(current); % Write joints to zero position
disp("Movement Complete")

figure(1)
q_desired = [q1_desired; q2_desired; q3_desired; q4_desired];
tau_all = tau_k;

figure(1)
for i = 1:4
    subplot(2,2,i)
    plot(t, q_desired(i,:), 'r.-')
    hold on
    plot(t, q_real(i,1:length(t)), 'b.-')
    xlabel('Time [s]')
    ylabel(['q', num2str(i), ' [rad]'])
    title(['Joint ', num2str(i), ' Angle'])
    legend('Desired', 'Current', 'Location', 'best')
    grid on
end

figure(2)
for i = 1:4
    subplot(2,2,i)
    plot(t, q_dot_real(i,1:length(t)), 'k.-')
    xlabel('Time [s]')
    ylabel(['dq', num2str(i), ' [rad/s]'])
    title(['Joint ', num2str(i), ' Velocity'])
    grid on
end

figure(3)
for i = 1:4
    subplot(2,2,i)
    plot(t, tau_all(i,:), 'm.-')
    xlabel('Time [s]')
    ylabel(['tau', num2str(i), ' [Nm]'])
    title(['Joint ', num2str(i), ' Control Torque'])
    grid on
end