%GENERATE_TRAJECTORY  Cartesian straight-line IK trajectory for OpenManipulator-X
%
%   [qd_traj, dqd_traj, ddqd_traj, t, ee_desired] = ...
%       generate_trajectory(q_top, q_bot, p, t_sample, tf, verbose)
%
%   INPUTS
%     q_top    (4x1) [rad]  Joint config at top of pole
%     q_bot    (4x1) [rad]  Joint config at bottom of pole
%     p               Robot parameter vector (from load_params)
%     t_sample        Sample period [s]          (default: 0.01)
%     tf              Total trajectory time [s]  (default: 10)
%     verbose         true/false — print tables and show plots (default: true)
%
%   OUTPUTS
%     qd_traj    (4 x N) [rad]       Desired joint positions
%     dqd_traj   (4 x N) [rad/s]     Desired joint velocities
%     ddqd_traj  (4 x N) [rad/s^2]   Desired joint accelerations
%     t          (1 x N) [s]         Time vector
%     ee_desired (3 x N) [m]         Straight-line Cartesian waypoints
function [qd_traj, dqd_traj, ddqd_traj, t, ee_desired] = generate_traj_NW(q_top, q_bot, p, t_sample, tf, verbose)

    % ── Defaults ─────────────────────────────────────────────────────────────
    if nargin < 4 || isempty(t_sample); t_sample = 0.01; end
    if nargin < 5 || isempty(tf);       tf       = 10.0; end
    if nargin < 6 || isempty(verbose);  verbose  = false; end
    
    % ── Time vector ─────────────────────────────────────────────────────────
    t = 0:t_sample:tf;
    N = length(t);
    
    % ── FK at endpoints ──────────────────────────────────────────────────────
    ee_top = FK_fun(q_top, p);
    ee_bot = FK_fun(q_bot, p);
    
    % ── IK self-test at known endpoints according to verbose keyword check
    if verbose
        fprintf('\n========== GENERATE_TRAJECTORY ==========\n');
        fprintf('=== FK Verification ===\n');
        fprintf('EE at q_top: x=%.4f  y=%.4f  z=%.4f [m]\n', ee_top);
        fprintf('EE at q_bot: x=%.4f  y=%.4f  z=%.4f [m]\n', ee_bot);
        fprintf('Delta Z = %.4f m\n\n', ee_bot(3)-ee_top(3));
    
        fprintf('=== IK Self-Test at Known Endpoints ===\n');
        q_sol_top = ik_newton(q_top(2:4), ee_top, p);
        q_sol_bot = ik_newton(q_bot(2:4), ee_bot, p);
        fprintf('q_top IK [deg]: q2=%7.3f  q3=%7.3f  q4=%7.3f\n', rad2deg(q_sol_top)');
        fprintf('  Expected    : q2=%7.3f  q3=%7.3f  q4=%7.3f\n\n', rad2deg(q_top(2:4))');
        fprintf('q_bot IK [deg]: q2=%7.3f  q3=%7.3f  q4=%7.3f\n', rad2deg(q_sol_bot)');
        fprintf('  Expected    : q2=%7.3f  q3=%7.3f  q4=%7.3f\n\n', rad2deg(q_bot(2:4))');
    end
    
    % ── s-curve profile: smooth start/stop ──────────────────────────────────
    %   s(t)  =  3(t/tf)^2 - 2(t/tf)^3    — zero velocity at t=0 and t=tf
    s_fun = @(tk) 3*(tk/tf).^2 - 2*(tk/tf).^3;
    
    % ── Pre-allocate ─────────────────────────────────────────────────────────
    qd_traj    = zeros(4, N);
    dqd_traj   = zeros(4, N);
    ddqd_traj  = zeros(4, N);
    ee_desired = zeros(3, N);
    ee_fk_check= zeros(3, N);
    
    % ── IK loop with warm-start ──────────────────────────────────────────────
    q_prev = q_top(2:4);
    
    if verbose; fprintf('Running IK for %d waypoints...\n', N); end
    
    for k = 1:N
        sk      = s_fun(t(k));
        ee_k    = ee_top + sk * (ee_bot - ee_top);
        ee_desired(:,k) = ee_k;
    
        q_sol = ik_newton(q_prev, ee_k, p);
    
        qd_traj(:,k)     = [0; q_sol];
        ee_fk_check(:,k) = FK_fun(qd_traj(:,k), p);
        q_prev = q_sol;
    end
    
    if verbose; fprintf('IK complete.\n\n'); end
    
    % ── Numerical derivatives (central differences) ──────────────────────────
    for k = 1:N
        if k == 1
            dqd_traj(:,k)  = (qd_traj(:,2)   - qd_traj(:,1))   / t_sample;
            ddqd_traj(:,k) = zeros(4,1);
        elseif k == N
            dqd_traj(:,k)  = (qd_traj(:,N)   - qd_traj(:,N-1)) / t_sample;
            ddqd_traj(:,k) = zeros(4,1);
        else
            dqd_traj(:,k)  = (qd_traj(:,k+1) - qd_traj(:,k-1)) / (2*t_sample);
            ddqd_traj(:,k) = (qd_traj(:,k+1) - 2*qd_traj(:,k)  + qd_traj(:,k-1)) / t_sample^2;
        end
    end

    % Verbose trigger: generates debug information such as figures and prints if set to true 
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
        fprintf('IK final q    [deg]: %7.3f  %7.3f  %7.3f  %7.3f\n', rad2deg(qd_traj(:,end))');
    
        fprintf('\n%-8s  %-8s  %-8s  %-8s  %-8s  |  %-8s  %-8s  %-8s\n', ...
            't[s]','q1[deg]','q2[deg]','q3[deg]','q4[deg]','EE_x[m]','EE_y[m]','EE_z[m]');
        fprintf('%s\n', repmat('-',1,80));
        for k = 1:10:N
            fprintf('%-8.2f  %-8.3f  %-8.3f  %-8.3f  %-8.3f  |  %-8.4f  %-8.4f  %-8.4f\n', ...
                t(k), rad2deg(qd_traj(1,k)), rad2deg(qd_traj(2,k)), ...
                rad2deg(qd_traj(3,k)), rad2deg(qd_traj(4,k)), ...
                ee_fk_check(1,k), ee_fk_check(2,k), ee_fk_check(3,k));
        end
    
        % All figures in one tiled window
        figure('Name','Trajectory Sanity Check','Position',[50 50 1400 900]);
    
        % Figure 1: Cartesian path sanity check
        subplot(1,2,1);
        plot(t, ee_desired(3,:)*100,  'b',  'LineWidth',2); hold on;
        plot(t, ee_fk_check(3,:)*100, 'r--','LineWidth',1.5);
        xlabel('Time [s]'); ylabel('z [cm]');
        legend('Desired','FK of IK joints','Location','best');
        title('Z-height — should overlap'); grid on;
    
        subplot(1,2,2);
        plot(t, ee_desired(1,:)*100,  'b',  'LineWidth',2); hold on;
        plot(t, ee_fk_check(1,:)*100, 'r--','LineWidth',1.5);
        xlabel('Time [s]'); ylabel('x [cm]');
        legend('Desired','FK of IK joints','Location','best');
        title('X-position — should stay constant'); grid on;
    
        % Figure 2: Joint trajectories
        jnames    = {'q_1','q_2','q_3','q_4'};
        q_top_deg = rad2deg(q_top);
        q_bot_deg = rad2deg(q_bot);
        for i = 1:4
            subplot(2,2,i);
            plot(t, rad2deg(qd_traj(i,:)), 'b', 'LineWidth',1.5); hold on;
            yline(q_top_deg(i), 'g--', 'q_{top}');
            yline(q_bot_deg(i), 'r--', 'q_{bot}');
            xlabel('Time [s]'); ylabel([jnames{i} ' [deg]']);
            title(['Joint ' num2str(i)]); grid on;
        end
    
        % Figure 3: Joint velocities
        for i = 1:4
            subplot(2,2,i);
            plot(t, rad2deg(dqd_traj(i,:)), 'LineWidth',1.5);
            xlabel('Time [s]'); ylabel([jnames{i} ' vel [deg/s]']);
            title(['Joint ' num2str(i) ' Velocity']); grid on;
        end
    
        % Figure 4: 3D EE path
        plot3(ee_fk_check(1,:)*100, ee_fk_check(2,:)*100, ee_fk_check(3,:)*100, ...
              'b', 'LineWidth',2); hold on;
        scatter3([ee_top(1),ee_bot(1)]*100, [ee_top(2),ee_bot(2)]*100, ...
                 [ee_top(3),ee_bot(3)]*100, 80, 'ro','filled');
        xlabel('x [cm]'); ylabel('y [cm]'); zlabel('z [cm]');
        title('3D End-Effector Path (should be vertical line)');
        legend('EE path','Start/End'); grid on; axis equal;
    end
end % generate_trajectory


% ── Local helper: Newton-Raphson IK ─────────────────────────────────────
function q_sol = ik_newton(q_init, ee_des, p, max_iter, tol)
    if nargin < 4; max_iter = 100; end
    if nargin < 5; tol      = 1e-10; end

    q = q_init;
    h = 1e-7;

    for iter = 1:max_iter
        ee_cur = FK_fun([0; q], p);

        F = [ee_cur(1) - ee_des(1);
             ee_cur(3) - ee_des(3);
             q(1) + q(2) + q(3)];

        if norm(F) < tol; break; end

        J = zeros(3,3);
        for j = 1:3
            dq     = zeros(3,1); dq(j) = h;
            ee_fwd = FK_fun([0; q+dq], p);
            ee_bwd = FK_fun([0; q-dq], p);
            J(1,j) = (ee_fwd(1) - ee_bwd(1)) / (2*h);
            J(2,j) = (ee_fwd(3) - ee_bwd(3)) / (2*h);
            J(3,j) = 1.0;
        end

        q = q - J \ F;
    end

    q_sol = q;
end