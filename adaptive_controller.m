function [tau, pi_hat_next] = adaptive_controller(p, q, qd, dq, dqd, ddqd, pi_hat, Kp, Kd, Gamma, dt)

    % Evaluate robot dynamics terms at current state

    e  = qd - q;
    de = dqd - dq;

    Y = Y_fun(q, dq, dqd, ddqd, p);

    % size(Y)

    tau = Y*pi_hat + Kp*e + Kd*de;

    pi_hat_next = pi_hat + Gamma*Y'*e*dt;
    % pi_hat_next = pi_hat;

    fprintf('tau: '); disp(tau)


end