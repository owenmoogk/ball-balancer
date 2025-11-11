% Trapezoidal trajectory planning in MATLAB (with automatic triangular fallback if needed)
clear; clc; close all;

% --- Parameters ---
xf = 10;          % target position [m]
amax = 2;         % maximum acceleration [m/s^2]

% --- Two cases with different vmax ---
vmax1 = 4.2;      % First case (will trigger triangular fallback)
vmax2 = 2;        % Second case (true trapezoidal with cruise phase)

% We'll store results in a struct for easy plotting
traj(1) = compute_trapezoidal(xf, amax, vmax1, 'b', 'Limited Speed (v_max = 4.2 m/s) - Triangular Profile');
traj(2) = compute_trapezoidal(xf, amax, vmax2, 'r', 'Conservative Speed (v_max = 2 m/s) - Full Trapezoidal with Cruise');

% --- Plotting ---
figure('Position',[100 100 900 600]);

subplot(2,1,1); hold on; grid on;
for i = 1:2
    plot(traj(i).t, traj(i).x, traj(i).color, 'LineWidth', 2, 'DisplayName', traj(i).label);
end
xlabel('Time [s]');
ylabel('Position [m]');
title('Position vs Time');
legend('Location','southeast');
xlim([0 max([traj.T])*1.05]);
ylim([0 xf*1.1]);

subplot(2,1,2); hold on; grid on;
for i = 1:2
    plot(traj(i).t, traj(i).v, traj(i).color, 'LineWidth', 2);
end
xlabel('Time [s]');
ylabel('Velocity [m/s]');
title('Velocity vs Time');
legend({traj.label}, 'Location','northeast');
xlim([0 max([traj.T])*1.05]);

% Print summary
fprintf('\n--- Summary ---\n');
for i = 1:2
    if traj(i).is_trapezoidal
        fprintf('%s: True trapezoidal, cruise time = %.3f s, total time = %.3f s\n', ...
            traj(i).label, traj(i).tc, traj(i).T);
    else
        fprintf('%s: Triangular fallback (vmax not reachable), vpeak = %.2f m/s, total time = %.3f s\n', ...
            traj(i).label, traj(i).vpeak, traj(i).T);
    end
end

% ==============================================================
% Helper function (place at the end of the script or in a separate file)
function traj = compute_trapezoidal(xf, amax, vmax, color, label)
    dt = 0.001;
    
    da = vmax^2 / (2*amax);
    dd = da;
    
    if (da + dd) <= xf
        % True trapezoidal with cruise
        dc = xf - da - dd;
        ta = vmax / amax;
        tc = dc / vmax;
        td = ta;
        T = ta + tc + td;
        vpeak = vmax;
        is_trapezoidal = true;
    else
        % Triangular fallback (vmax not reachable)
        vpeak = sqrt(amax * xf);
        ta = vpeak / amax;
        td = ta;
        T = ta + td;
        tc = 0;
        dc = 0;
        is_trapezoidal = false;
    end
    
    t = 0:dt:T;
    x = zeros(size(t));
    v = zeros(size(t));
    
    for i = 1:length(t)
        ti = t(i);
        
        if ti <= ta
            % Acceleration phase
            v(i) = amax * ti;
            x(i) = 0.5 * amax * ti^2;
            
        elseif tc > 0 && ti <= (ta + tc)
            % Cruise phase
            v(i) = vmax;
            x(i) = da + vmax * (ti - ta);
            
        else
            % Deceleration phase
            tdec = ti - (ta + tc);
            v(i) = vpeak - amax * tdec;
            x(i) = xf - dd + vpeak * tdec - 0.5 * amax * tdec^2;
        end
    end
    
    % Store results
    traj.t = t;
    traj.x = x;
    traj.v = v;
    traj.T = T;
    traj.vpeak = vpeak;
    traj.tc = tc;
    traj.is_trapezoidal = is_trapezoidal;
    traj.color = color;
    traj.label = label;
end