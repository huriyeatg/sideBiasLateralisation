% x: behavior times; y: two-photon times
x = reward_echo_behav(:);
y = reward_echo_twoP(:);

thresh_ms = 20;      % stop when max |residual| < 20 ms
maxIter   = 10;
iter      = 0;

% start with no correction
y_hat     = x;                       % predicted y from affine map(s)
signal    = (y - y_hat)*1000;        % residuals in ms
variance  = max(abs(signal));
delay_array = zeros(size(x));        % per-event correction in seconds

while variance > thresh_ms && iter < maxIter
    disp(iter)
    iter = iter + 1;

    % find step changes on current residuals (either sign)
    jump_idx = find(abs(diff(signal)) > thresh_ms);
    edges    = [0; jump_idx(:); numel(x)];
    
    % piecewise AFFINE fit: y ≈ a + b*x for each segment
    for k = 1:numel(edges)-1
        idx = (edges(k)+1):edges(k+1);
        xx = x(idx);  yy = y(idx);

        % robust linear fit; robustfit returns [a; b] s.t. yy ≈ a + b*xx
        ab = robustfit(xx, yy);          % Statistics TB; else use polyfit
        a  = ab(1);
        b  = ab(2);

        % predicted y for this segment
        y_hat(idx) = a + b*xx;

        % per-event correction to apply to behavior time (seconds)
        % (how much to shift x so it matches y): a + b*x - x
        delay_array(idx) = (a + b*xx) - xx;
    end

    % recompute residuals & check convergence
    signal_new = (y - y_hat)*1000;          % ms
    new_var    = max(abs(signal_new));

    % stop if not improving (prevents oscillation)
    if new_var >= variance
        break;
    end

    signal   = signal_new;
    variance = new_var;
end