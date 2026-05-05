%GENERATE_TRAJECTORY  Cartesian straight-line IK trajectory for OpenManipulator-X
%
%   Computes joint-space position, velocity, and acceleration analytically:
%     - Position:     IK solved at each waypoint along s-curve Cartesian path
%                     q4 interpolated from q_top(4) to q_bot(4) via same s-curve
%     - Velocity:     q_dot  = J^{-1} * ee_dot       (first-order kinematics)
%                     q4_dot = s_dot * (q_bot(4) - q_top(4))   (analytical)
%     - Acceleration: q_ddot = J^{-1} * (ee_ddot - Jdot * q_dot)
%                     q4_ddot = s_ddot * (q_bot(4) - q_top(4)) (analytical)
%
%   The s-curve profile and its derivatives are evaluated analytically:
%     s(t)      =  3*(t/tf)^2 - 2*(t/tf)^3
%     s_dot(t)  =  6*t/tf^2  - 6*t^2/tf^3
%     s_ddot(t) =  6/tf^2    - 12*t/tf^3
%
%   [qd_traj, dqd_traj, ddqd_traj, t, ee_desired] = ...
%       generate_trajectory(q_top, q_bot, p, t_sample, tf, verbose)
%
%   INPUTS
%     q_top    (4x1) [rad]  Joint config at top of stroke
%     q_bot    (4x1) [rad]  Joint config at bottom of stroke
%     p               Robot parameter vector (from identification_result.mat)
%     t_sample        Sample period [s]          (default: 0.01)
%     tf              Total trajectory time [s]  (default: 10)
%     verbose         true/false — print tables and show plots (default: false)
%
%   OUTPUTS
%     qd_traj    (4 x N) [rad]       Desired joint positions
%     dqd_traj   (4 x N) [rad/s]     Desired joint velocities   (analytical)
%     ddqd_traj  (4 x N) [rad/s^2]   Desired joint accelerations (analytical)
%     t          (1 x N) [s]         Time vector
%     ee_desired (3 x N) [m]         Straight-line Cartesian waypoints

function [qd_traj, dqd_traj, ddqd_traj, t, ee_desired] = ...
        generate_trajectory(q_top, q_bot, p, t_sample, tf, verbose)

    %% ── Defaults ────────────────────────────────────────────────────────────
    if nargin < 4 || isempty(t_sample); t_sample = 0.01; end
    if nargin < 5 || isempty(tf);       tf       = 10.0; end
    if nargin < 6 || isempty(verbose);  verbose  = false; end

    %% ── Time vector ─────────────────────────────────────────────────────────
    t = 0:t_sample:tf;
    N = length(t);

    %% ── FK at endpoints ─────────────────────────────────────────────────────
    ee_top = FK_fun(q_top, p);
    ee_bot = FK_fun(q_bot, p);
    delta_ee = ee_bot - ee_top;     % Cartesian displacement vector [3x1]

    %% ── Verbose: FK verification and IK self-test ───────────────────────────
    if verbose
        fprintf('\n========== GENERATE_TRAJECTORY (Analytical) ==========\n');
        fprintf('=== FK Verification ===\n');
        fprintf('EE at q_top: x=%.4f  y=%.4f  z=%.4f [m]\n', ee_top);
        fprintf('EE at q_bot: x=%.4f  y=%.4f  z=%.4f [m]\n', ee_bot);
        fprintf('Delta Z = %.4f m\n\n', delta_ee(3));

        fprintf('=== IK Self-Test at Known Endpoints ===\n');
        q_sol_top  = ik_newton(q_top(2:4),  ee_top, p, q_top(4));
        q_sol_bot  = ik_newton(q_bot(2:4),  ee_bot, p, q_bot(4));
        fprintf('q_top IK [deg]: q2=%7.3f  q3=%7.3f  q4=%7.3f\n', rad2deg(q_sol_top)');
        fprintf('  Expected    : q2=%7.3f  q3=%7.3f  q4=%7.3f\n\n', rad2deg(q_top(2:4))');
        fprintf('q_bot IK [deg]: q2=%7.3f  q3=%7.3f  q4=%7.3f\n', rad2deg(q_sol_bot)');
        fprintf('  Expected    : q2=%7.3f  q3=%7.3f  q4=%7.3f\n\n', rad2deg(q_bot(2:4))');
    end

    %% ── S-curve profile (analytical) ────────────────────────────────────────
    %   Position:     s      =  3*(t/tf)^2 - 2*(t/tf)^3        ∈ [0,1]
    %   Velocity:     s_dot  =  6*t/tf^2   - 6*t^2/tf^3        zero at t=0,tf
    %   Acceleration: s_ddot =  6/tf^2     - 12*t/tf^3         zero at t=0,tf
    tau       = t / tf;                         % normalized time [0,1]
    s         =  3*tau.^2  - 2*tau.^3;
    s_dot     =  6*tau/tf  - 6*tau.^2/tf;       % ds/dt
    s_ddot    =  6/tf^2    - 12*tau/tf^2;        % d²s/dt²

    %% ── Pre-allocate ────────────────────────────────────────────────────────
    qd_traj    = zeros(4, N);
    dqd_traj   = zeros(4, N);
    ddqd_traj  = zeros(4, N);
    ee_desired = zeros(3, N);
    ee_fk_check= zeros(3, N);

    %% ── q4 interpolation (analytical, same s-curve) ─────────────────────────
    % q4 travels from q_top(4) to q_bot(4) using the same s-curve profile.
    % This ensures qd_traj(:,end) == q_bot exactly, and q4_dot/q4_ddot
    % are zero at both endpoints (smooth start and stop).
    delta_q4 = q_bot(4) - q_top(4);
    q4_traj  = q_top(4) + s      * delta_q4;   % [1 x N] position
    q4_dot   =            s_dot  * delta_q4;   % [1 x N] velocity
    q4_ddot  =            s_ddot * delta_q4;   % [1 x N] acceleration

    %% ── IK loop with warm-start ─────────────────────────────────────────────
    q_prev   = q_top(2:4);

    if verbose; fprintf('Running IK for %d waypoints...\n', N); end

    for k = 1:N
        %% Cartesian waypoint
        ee_k               = ee_top + s(k) * delta_ee;
        ee_desired(:,k)    = ee_k;

        %% IK: solve q2,q3 with q4 fixed to interpolated value at this step
        q_sol              = ik_newton(q_prev, ee_k, p, q4_traj(k));
        qd_traj(:,k)       = [0; q_sol];
        ee_fk_check(:,k)   = FK_fun(qd_traj(:,k), p);
        q_prev             = q_sol;

        %% ── Analytical velocity: q_dot = J^{-1} * ee_dot ───────────────────
        %   Cartesian velocity at this waypoint:
        %     ee_dot = s_dot(k) * delta_ee
        ee_dot_k = s_dot(k) * delta_ee;        % [3x1] m/s

        %   Numerical Jacobian (x and z rows only, columns q2 q3 q4)
        %   J is 2x3: maps [dq2; dq3; dq4] → [dx; dz]
        J = numerical_jacobian(qd_traj(:,k), p);    % [2x3]

        %   Constrain: q4 is fixed => dq4/dt = 0
        %   Solve 2x2 system for dq2/dt, dq3/dt
        J23     = J(:, 1:2);                    % [2x2] columns for q2,q3
        ee_dot_xz = [ee_dot_k(1); ee_dot_k(3)];% [2x1] x and z components

        if rcond(J23) > 1e-8
            qdot_23 = J23 \ ee_dot_xz;         % [2x1]
        else
            qdot_23 = zeros(2,1);               % at singularity: zero velocity
        end

        dqd_traj(:,k) = [0; qdot_23; q4_dot(k)];   % q1_dot=0, q4_dot analytical

        %% ── Analytical acceleration: q_ddot = J^{-1}*(ee_ddot - Jdot*q_dot) 
        %   Cartesian acceleration:
        %     ee_ddot = s_ddot(k) * delta_ee
        ee_ddot_k    = s_ddot(k) * delta_ee;
        ee_ddot_xz   = [ee_ddot_k(1); ee_ddot_k(3)];

        %   Jdot approximated by finite difference of J at adjacent steps
        if k == 1 || k == N
            Jdot23 = zeros(2,2);
        else
            J_prev = numerical_jacobian(qd_traj(:,max(k-1,1)), p);
            J_next = numerical_jacobian(qd_traj(:,min(k+1,N)), p);
            Jdot23 = (J_next(:,1:2) - J_prev(:,1:2)) / (2*t_sample);
        end

        if rcond(J23) > 1e-8
            ddqd_traj(2:3,k) = J23 \ (ee_ddot_xz - Jdot23 * qdot_23);
        else
            ddqd_traj(2:3,k) = zeros(2,1);
        end
        ddqd_traj(4,k) = q4_ddot(k);   % q4_ddot analytical, q1_ddot=0
    end

    if verbose; fprintf('IK + analytical derivatives complete.\n\n'); end

    %% ── Verbose diagnostics ─────────────────────────────────────────────────
    if verbose
        fprintf('=== IK Self-Consistency: FK(IK(ee)) vs ee_desired ===\n');
        fprintf('%-6s  %-12s  %-12s  %-12s\n','t[s]','err_x[mm]','err_y[mm]','err_z[mm]');
        fprintf('%s\n', repmat('-',1,50));
        for k = 1:10:N
            err = (ee_fk_check(:,k) - ee_desired(:,k)) * 1000;
            fprintf('%-6.1f  %-12.5f  %-12.5f  %-12.5f\n', t(k), err(1), err(2), err(3));
        end

        fprintf('\n=== Endpoint Check ===\n');
        fprintf('Expected q_bot [deg]: %7.3f  %7.3f  %7.3f  %7.3f\n', rad2deg(q_bot)');
        fprintf('IK final q    [deg]: %7.3f  %7.3f  %7.3f  %7.3f\n\n', rad2deg(qd_traj(:,end))');

        fprintf('%-8s  %-8s  %-8s  %-8s  %-8s  |  %-8s  %-8s  %-8s\n', ...
            't[s]','q1[deg]','q2[deg]','q3[deg]','q4[deg]','EE_x[m]','EE_y[m]','EE_z[m]');
        fprintf('%s\n', repmat('-',1,80));
        for k = 1:10:N
            fprintf('%-8.2f  %-8.3f  %-8.3f  %-8.3f  %-8.3f  |  %-8.4f  %-8.4f  %-8.4f\n', ...
                t(k), rad2deg(qd_traj(1,k)), rad2deg(qd_traj(2,k)), ...
                rad2deg(qd_traj(3,k)), rad2deg(qd_traj(4,k)), ...
                ee_fk_check(1,k), ee_fk_check(2,k), ee_fk_check(3,k));
        end

        %% Sanity plots
        figure('Name','Trajectory Sanity Check','Position',[50 50 1400 900]);

        subplot(2,3,1)
        plot(t, ee_desired(3,:)*100,  'b',  'LineWidth',2); hold on
        plot(t, ee_fk_check(3,:)*100, 'r--','LineWidth',1.5)
        xlabel('Time [s]'); ylabel('z [cm]')
        legend('Desired','FK of IK joints'); title('Z-height'); grid on

        subplot(2,3,2)
        plot(t, ee_desired(1,:)*100,  'b',  'LineWidth',2); hold on
        plot(t, ee_fk_check(1,:)*100, 'r--','LineWidth',1.5)
        xlabel('Time [s]'); ylabel('x [cm]')
        legend('Desired','FK of IK joints'); title('X-position (constant)'); grid on

        subplot(2,3,3)
        plot3(ee_fk_check(1,:)*100, ee_fk_check(2,:)*100, ee_fk_check(3,:)*100, ...
              'b','LineWidth',2); hold on
        scatter3([ee_top(1),ee_bot(1)]*100,[ee_top(2),ee_bot(2)]*100, ...
                 [ee_top(3),ee_bot(3)]*100, 80,'ro','filled')
        xlabel('x [cm]'); ylabel('y [cm]'); zlabel('z [cm]')
        title('3D EE Path (vertical line)'); grid on; axis equal

        jnames = {'q_1','q_2','q_3','q_4'};
        for i = 1:4
            subplot(4,3, (i-1)*3 + 1 + 3*(i>1))  % offset layout
        end

        figure('Name','Trajectory Signals','Position',[50 50 1400 700])
        for i = 1:4
            subplot(3,4,i)
            plot(t, rad2deg(qd_traj(i,:)), 'b','LineWidth',1.5)
            xlabel('Time [s]'); ylabel([jnames{i},' [deg]'])
            title(['q',num2str(i),' position']); grid on

            subplot(3,4,4+i)
            plot(t, rad2deg(dqd_traj(i,:)), 'r','LineWidth',1.5)
            xlabel('Time [s]'); ylabel([jnames{i},' [deg/s]'])
            title(['q',num2str(i),' velocity']); grid on

            subplot(3,4,8+i)
            plot(t, rad2deg(ddqd_traj(i,:)), 'k','LineWidth',1.5)
            xlabel('Time [s]'); ylabel([jnames{i},' [deg/s^2]'])
            title(['q',num2str(i),' acceleration']); grid on
        end
        sgtitle('Trajectory Reference Signals (Analytical Derivatives)')
    end

end % generate_trajectory


%% ── Local: Newton-Raphson IK ────────────────────────────────────────────────
function q_sol = ik_newton(q_init, ee_des, p, q4_fixed, max_iter, tol)
    if nargin < 5; max_iter = 100;   end
    if nargin < 6; tol      = 1e-10; end

    q = q_init;     % [q2; q3; q4]
    h = 1e-7;

    for iter = 1:max_iter
        ee_cur = FK_fun([0; q], p);

        % Residual: match x, match z, hold q4 fixed
        F = [ee_cur(1) - ee_des(1);
             ee_cur(3) - ee_des(3);
             q(3)      - q4_fixed ];

        if norm(F) < tol; break; end

        % Numerical Jacobian of F w.r.t. [q2; q3; q4]
        J = zeros(3,3);
        for j = 1:3
            dq       = zeros(3,1); dq(j) = h;
            ee_fwd   = FK_fun([0; q+dq], p);
            ee_bwd   = FK_fun([0; q-dq], p);
            J(1,j)   = (ee_fwd(1) - ee_bwd(1)) / (2*h);   % dx/dqj
            J(2,j)   = (ee_fwd(3) - ee_bwd(3)) / (2*h);   % dz/dqj
            J(3,j)   = (j == 3);                            % dq4/dq4 = 1
        end

        q = q - J \ F;
    end

    q_sol = q;
end


%% ── Local: Numerical Jacobian [dx/dq2, dx/dq3; dz/dq2, dz/dq3] ────────────
function J = numerical_jacobian(q_full, p)
% Returns the 2x3 Jacobian mapping [dq2; dq3; dq4] → [dx; dz]
% J(1,:) = dx/d[q2,q3,q4]
% J(2,:) = dz/d[q2,q3,q4]

    h  = 1e-7;
    J  = zeros(2, 3);

    for j = 2:4     % joints q2, q3, q4 (indices 2,3,4 in q_full)
        dq          = zeros(4,1); dq(j) = h;
        ee_fwd      = FK_fun(q_full + dq, p);
        ee_bwd      = FK_fun(q_full - dq, p);
        col         = j - 1;                        % column index 1,2,3
        J(1, col)   = (ee_fwd(1) - ee_bwd(1)) / (2*h);  % dx/dqj
        J(2, col)   = (ee_fwd(3) - ee_bwd(3)) / (2*h);  % dz/dqj
    end
end