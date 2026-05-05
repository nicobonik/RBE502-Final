function [tau, pi_hat_next] = adaptive_controller(p, q, qd, dq, dqd, ddqd, pi_hat, Kd, Lambda, Gamma, dt)

    % Evaluate robot dynamics terms at current state

    e  = qd - q;
    de = dqd - dq;
    %% Filtered error s = e_dot + Lambda * e  (Siciliano eq. 8.95)
    s = de + Lambda * e;

    %% Reference velocity and acceleration  (Siciliano eq. 8.94)
    dq_r  = dqd  + Lambda * e;     % q_dot_r
    ddq_r = ddqd + Lambda * de;    % q_ddot_r

    %% Regressor evaluated at reference trajectory  (Siciliano eq. 8.96)
    % Y maps parameter vector pi to inertial torques:
    %   Y(q, dq, dq_r, ddq_r) * pi = M(q)*ddq_r + C(q,dq)*dq_r + G(q)
    Y = Y_fun(q, dq, dq_r, ddq_r, p);

    %% Control law  (Siciliano eq. 8.97)
    % No separate Kp*e term — position error is embedded in s via Lambda
    tau = Y * pi_hat + Kd * s;

    %% Adaptation law — Euler integration  (Siciliano eq. 8.98)
    pi_hat_next = pi_hat + Gamma * Y' * s * dt;

    disp(tau);


end