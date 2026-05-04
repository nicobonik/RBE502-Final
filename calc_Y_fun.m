p_sym    = sym('p',    [23 1], 'real');
q_sym    = sym('q',    [4  1], 'real');
dq_sym   = sym('dq',   [4  1], 'real');
dqr_sym  = sym('dqr',  [4  1], 'real');
ddqr_sym = sym('ddqr', [4  1], 'real');

tau_sym = M_fun(q_sym, p_sym)*ddqr_sym ...
        + C_fun(q_sym, dq_sym, p_sym)*dqr_sym ...
        + G_fun(q_sym, p_sym);

% Only differentiate w.r.t. inertial params
Y_sym = jacobian(tau_sym, p_sym(7:22));   % 4x16

% Pass full p_sym so geometric params p(1:6) and g p(23) are correct
Y_fun = matlabFunction(Y_sym, ...
    'Vars', {q_sym, dq_sym, dqr_sym, ddqr_sym, p_sym}, ...
    'File', 'Y_fun.m');